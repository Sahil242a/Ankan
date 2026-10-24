"""TrOCR engine against a tiny, randomly initialised VisionEncoderDecoder model.

No weights are downloaded. Random weights produce gibberish text, but the mechanics are
real: batching, greedy generation, padding after end-of-sequence, and the confidence
computed from transition scores. The confidence is checked against an independent
teacher-forced recomputation of the same token probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from ankan.ocr import TrOCREngine  # noqa: E402
from ankan.ocr.preprocess import LineCrop, deskew  # noqa: E402
from ankan.schemas import BoundingBox  # noqa: E402
from ankan.synthetic import render_sheet  # noqa: E402

BOS, PAD, EOS = 0, 1, 2
VOCAB = ["<s>", "<pad>", "</s>", "<unk>", "leaf", "light", "water"]


class TinyProcessor:
    """Exposes exactly the TrOCRProcessor surface the engine uses."""

    def __init__(self) -> None:
        tok = tokenizers.Tokenizer(
            tokenizers.models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>")
        )
        tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        self.tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=tok,
            bos_token="<s>",
            pad_token="<pad>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        self.image_processor = transformers.ViTImageProcessor(size={"height": 32, "width": 32})

    def __call__(self, images: Any, return_tensors: str) -> Any:
        return self.image_processor(images=images, return_tensors=return_tensors)

    def batch_decode(self, sequences: Any, skip_special_tokens: bool) -> list[str]:
        return list(self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens))


def tiny_model(seed: int) -> Any:
    torch.manual_seed(seed)
    encoder = transformers.ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    decoder = transformers.TrOCRConfig(
        vocab_size=len(VOCAB),
        d_model=32,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=64,
        max_position_embeddings=64,
        pad_token_id=PAD,
        bos_token_id=BOS,
        eos_token_id=EOS,
        decoder_start_token_id=EOS,
    )
    config = transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
    config.decoder_start_token_id = EOS
    config.pad_token_id = PAD
    config.eos_token_id = EOS
    model = transformers.VisionEncoderDecoderModel(config=config)
    model.generation_config.decoder_start_token_id = EOS
    model.generation_config.pad_token_id = PAD
    model.generation_config.eos_token_id = EOS
    return model


def crops(count: int) -> list[LineCrop]:
    rng = np.random.default_rng(0)
    return [
        LineCrop(
            box=BoundingBox(x=0, y=0, width=120, height=30),
            image=rng.integers(0, 256, size=(30, 120, 3), dtype=np.uint8),
        )
        for _ in range(count)
    ]


def reference_confidence(engine: TrOCREngine, crop: LineCrop) -> tuple[str, float]:
    """Greedy-decode one line, then score the chosen tokens with a plain forward pass."""
    from PIL import Image as PILImage

    model, processor = engine._model, engine._processor
    pixels = processor(images=[PILImage.fromarray(crop.image)], return_tensors="pt").pixel_values
    with torch.inference_mode():
        sequence = model.generate(
            pixels, max_new_tokens=engine.max_new_tokens, num_beams=1, do_sample=False
        )
        logits = model(pixel_values=pixels, decoder_input_ids=sequence[:, :-1]).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    targets = sequence[:, 1:]
    chosen = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    chosen = chosen[targets[0] != PAD]
    text = processor.batch_decode(sequence, skip_special_tokens=True)[0].strip()
    return text, float(torch.exp(chosen.mean()))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_batched_confidence_matches_teacher_forced_reference(seed: int) -> None:
    engine = TrOCREngine(
        processor=TinyProcessor(),
        model=tiny_model(seed),
        device="cpu",
        batch_size=3,
        max_new_tokens=12,
    )
    lines = crops(5)  # two batches (3 + 2), so padding inside a batch is exercised
    results = engine.recognize(lines)
    assert len(results) == 5
    for crop, (text, confidence) in zip(lines, results, strict=True):
        expected_text, expected_confidence = reference_confidence(engine, crop)
        assert text == expected_text
        assert 0.0 < confidence <= 1.0
        assert confidence == pytest.approx(expected_confidence, rel=1e-4)


class EndRowsAt:
    """Wraps the real model; `generate` forces row i to emit end-of-sequence at step stops[i].

    Generation, padding of finished rows, and transition scores are all the library's own;
    only *when* each row ends is controlled, so a batch is guaranteed to contain padding.
    """

    def __init__(self, model: Any, stops: list[int]) -> None:
        self.model = model
        self.stops = stops

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        stops = self.stops

        class Force(transformers.LogitsProcessor):  # type: ignore[misc]
            def __call__(self, input_ids: Any, scores: Any) -> Any:
                step = input_ids.shape[1] - 1  # tokens generated so far
                for row, stop in enumerate(stops[: scores.shape[0]]):
                    if step == stop:
                        scores[row] = float("-inf")
                        scores[row, EOS] = 0.0
                    elif step < stop:
                        scores[row, EOS] = float("-inf")
                    # After the stop the row is finished: the library pads it, and its scores
                    # stay finite model scores, exactly as in normal generation.
                return scores

        processors = transformers.LogitsProcessorList([Force()])
        return self.model.generate(*args, logits_processor=processors, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def test_padding_after_end_of_sequence_is_ignored() -> None:
    """Rows that end early are padded; each row's result must equal running it alone."""
    stops = [2, 6, 3, 9]
    wrapper = EndRowsAt(tiny_model(0), stops)
    engine = TrOCREngine(
        processor=TinyProcessor(), model=wrapper, device="cpu", batch_size=4, max_new_tokens=12
    )
    lines = crops(4)
    batched = engine.recognize(lines)

    # The batch really contains padding: rows that stop early are filled with PAD.
    from PIL import Image as PILImage

    pixels = engine._processor(
        images=[PILImage.fromarray(c.image) for c in lines], return_tensors="pt"
    ).pixel_values
    with torch.inference_mode():
        out = wrapper.generate(
            pixels,
            max_new_tokens=12,
            num_beams=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        logprobs = wrapper.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )
    generated = out.sequences[:, -logprobs.shape[1] :]
    assert [int((row != PAD).sum()) for row in generated] == [s + 1 for s in stops]  # + EOS
    # Without masking, the padded row's confidence would be different: the test bites.
    naive = float(torch.exp(logprobs[0][torch.isfinite(logprobs[0])].mean()))
    assert naive != pytest.approx(batched[0][1], rel=1e-3)

    for index, crop in enumerate(lines):
        wrapper.stops = [stops[index]]
        solo_text, solo_conf = engine.recognize([crop])[0]
        text, confidence = batched[index]
        assert text == solo_text
        assert confidence == pytest.approx(solo_conf, rel=1e-5)


def test_read_full_page() -> None:
    page, _ = deskew(render_sheet(["first line of writing", "second line", "third"], angle=2))
    engine = TrOCREngine(processor=TinyProcessor(), model=tiny_model(0), device="cpu")
    result = engine.read(page)
    assert result.engine == "trocr"
    assert len(result.lines) == 3
    assert all(line.box is not None for line in result.lines)
    assert all(line.confidence is not None and 0 <= line.confidence <= 1 for line in result.lines)
    assert engine.recognize([]) == []


def test_model_and_processor_are_injected_together() -> None:
    with pytest.raises(ValueError, match="both"):
        TrOCREngine(model=tiny_model(0))


def test_failed_model_download_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from ankan.errors import OCRError

    def offline(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("proxy refused the connection")  # not an OSError subclass path

    monkeypatch.setattr(transformers.TrOCRProcessor, "from_pretrained", offline)
    with pytest.raises(OCRError, match="check the internet connection"):
        TrOCREngine("microsoft/trocr-base-handwritten").recognize(crops(1))


def test_fast_mode_quantizes_and_reads_the_same_way() -> None:
    lines = crops(5)
    full = TrOCREngine(processor=TinyProcessor(), model=tiny_model(0), device="cpu")
    fast = TrOCREngine(processor=TinyProcessor(), model=tiny_model(0), device="cpu", fast=True)
    assert any("quantized" in type(m).__module__ for m in fast._model.modules())
    assert not any("quantized" in type(m).__module__ for m in full._model.modules())
    results = fast.recognize(lines)
    assert len(results) == 5
    assert all(0.0 <= confidence <= 1.0 for _, confidence in results)


def test_fast_mode_falls_back_when_quantization_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no quantized engine on this platform")

    monkeypatch.setattr(torch.ao.quantization, "quantize_dynamic", broken)
    engine = TrOCREngine(processor=TinyProcessor(), model=tiny_model(0), device="cpu", fast=True)
    assert len(engine.recognize(crops(2))) == 2  # still works, at full precision


def test_one_batch_holds_a_whole_page() -> None:
    assert TrOCREngine().batch_size >= 16


def test_warm_up_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_load(name: str) -> Any:
        calls.append(name)
        return TinyProcessor() if len(calls) == 1 else tiny_model(0)

    monkeypatch.setattr(transformers.TrOCRProcessor, "from_pretrained", fake_load)
    monkeypatch.setattr(transformers.VisionEncoderDecoderModel, "from_pretrained", fake_load)
    engine = TrOCREngine("some/model", device="cpu")
    engine.warm_up()
    engine.warm_up()
    engine.recognize(crops(1))
    assert len(calls) == 2  # processor + model, loaded a single time
