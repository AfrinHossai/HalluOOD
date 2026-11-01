"""Lightweight adapters for frozen image-question-answering models.

The recommended local checkpoint is HuggingFaceTB/SmolVLM-500M-Instruct.
Transformers and PyTorch are optional runtime dependencies and are imported only
when ``TransformersVLMAdapter.from_pretrained`` is called.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from halluood.uncertainty import CandidateSequenceLogProbability

RECOMMENDED_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
RECOMMENDED_MODEL_REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"

ImageInput = str | Path | Image.Image
PipelineCallable = Callable[..., Any]


class ModelAdapterError(RuntimeError):
    """Raised when a model adapter cannot load or use its backend."""


@dataclass(frozen=True)
class GenerationSettings:
    """Settings for repeatable greedy generation."""

    max_new_tokens: int = 32
    seed: int = 2027
    num_beams: int = 1
    do_sample: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.num_beams, bool)
            or not isinstance(self.num_beams, int)
            or self.num_beams < 1
        ):
            raise ValueError("num_beams must be a positive integer")
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be a boolean")
        if self.do_sample:
            raise ValueError("this adapter currently supports deterministic greedy decoding only")


@dataclass(frozen=True)
class SamplingSettings:
    """Controlled settings for repeated stochastic answer generation."""

    num_generations: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 2027
    max_new_tokens: int = 32

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_generations, bool)
            or not isinstance(self.num_generations, int)
            or self.num_generations < 1
        ):
            raise ValueError("num_generations must be a positive integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, Real)
            or not isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be a finite number greater than zero")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, Real)
            or not isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be a finite number in (0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if self.seed + self.num_generations - 1 > 2**63 - 1:
            raise ValueError("seed range exceeds the supported torch seed range")

    @property
    def generation_seeds(self) -> tuple[int, ...]:
        """Return one distinct reproducible seed per response."""

        return tuple(self.seed + index for index in range(self.num_generations))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable sampling settings and the per-response seeds."""

        return {
            "do_sample": True,
            "num_beams": 1,
            "num_generations": self.num_generations,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "seed": self.seed,
            "generation_seeds": list(self.generation_seeds),
            "seed_strategy": "generation_seed = base_seed + zero_based_generation_index",
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class ModelMetadata:
    """Model, runtime, and decoding details to store with an inference run."""

    model_id: str
    requested_revision: str | None
    resolved_revision: str | None
    backend: str
    device: str
    precision: str | None
    transformers_version: str | None
    torch_version: str | None
    generation: GenerationSettings = field(default_factory=GenerationSettings)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata for a run manifest."""

        return asdict(self)


class VLMAdapter(Protocol):
    """Shared interface for a model that answers a question about an image."""

    @property
    def metadata(self) -> ModelMetadata: ...

    def generate_answer(self, image: ImageInput, question: str) -> str: ...

    def generate_sampled_answers(
        self,
        image: ImageInput,
        question: str,
        settings: SamplingSettings,
    ) -> list[str]: ...


def _validate_question(question: str) -> None:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")


def _load_rgb_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not isinstance(image, (str, Path)):
        raise TypeError("image must be a local path or a PIL image")

    try:
        with Image.open(image) as opened:
            opened.load()
            return opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"could not read image {image!s}: {exc}") from exc


def _extract_text(value: Any) -> str:
    """Handle the string and chat-message forms returned by Transformers."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        assistant_messages = [
            item
            for item in value
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        if assistant_messages:
            return _extract_text(assistant_messages[-1])
        if value:
            return _extract_text(value[-1])
        return ""
    if isinstance(value, dict):
        for key in ("generated_text", "text"):
            if key in value:
                return _extract_text(value[key])
        if "content" in value:
            content = value["content"]
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "".join(text_parts).strip()
            return _extract_text(content)
    raise ModelAdapterError(
        f"could not extract generated answer from output of type {type(value).__name__}"
    )


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


class TransformersVLMAdapter:
    """Use a Transformers image-text-to-text pipeline for local inference.

    Constructing this adapter with a preloaded pipeline performs no downloads.
    ``from_pretrained`` is the explicit opt-in that loads a model checkpoint.
    """

    def __init__(
        self,
        pipeline: PipelineCallable,
        *,
        model_id: str,
        requested_revision: str | None,
        resolved_revision: str | None = None,
        device: str = "cpu",
        precision: str | None = None,
        transformers_version: str | None = None,
        torch_version: str | None = None,
        generation: GenerationSettings | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if not callable(pipeline):
            raise TypeError("pipeline must be callable")
        self._pipeline = pipeline
        self._generation = generation or GenerationSettings()
        self._torch = torch_module
        self._metadata = ModelMetadata(
            model_id=model_id,
            requested_revision=requested_revision,
            resolved_revision=resolved_revision,
            backend="transformers.image-text-to-text",
            device=device,
            precision=precision,
            transformers_version=transformers_version,
            torch_version=torch_version,
            generation=self._generation,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = RECOMMENDED_MODEL_ID,
        *,
        revision: str = RECOMMENDED_MODEL_REVISION,
        device: str | None = None,
        generation: GenerationSettings | None = None,
        cache_dir: str | Path | None = None,
    ) -> TransformersVLMAdapter:
        """Load a pinned checkpoint; this method may download model weights."""

        resolved_cache_dir = None
        if cache_dir is not None:
            resolved_cache_dir = Path(cache_dir).expanduser()
            resolved_cache_dir.mkdir(parents=True, exist_ok=True)
            # Transformers uses model_kwargs for weights and HF_HOME for the
            # processor assets loaded by pipeline(). Set this before importing
            # Hugging Face libraries so both use the requested local cache.
            os.environ["HF_HOME"] = str(resolved_cache_dir)

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ModelAdapterError(
                "pretrained inference requires compatible PyTorch and Transformers installs; "
                "this method has not installed them or downloaded model weights"
            ) from exc

        resolved_device = device
        if resolved_device is None:
            mps_backend = getattr(torch.backends, "mps", None)
            resolved_device = "mps" if mps_backend and mps_backend.is_available() else "cpu"

        pipeline_kwargs: dict[str, Any] = {
            "model": model_id,
            "revision": revision,
            "device": resolved_device,
        }
        if resolved_cache_dir is not None:
            pipeline_kwargs["model_kwargs"] = {"cache_dir": str(resolved_cache_dir)}
        if resolved_device == "mps":
            # SmolVLM's published config defaults to bfloat16. Float16 is the
            # safer broadly supported MPS inference dtype for this pilot.
            pipeline_kwargs["dtype"] = torch.float16
        elif resolved_device == "cpu":
            # Keep CPU inference compatible with backends lacking bfloat16
            # kernels; the 500M checkpoint still fits comfortably in RAM.
            pipeline_kwargs["dtype"] = torch.float32

        loaded_pipeline = pipeline(
            "image-text-to-text",
            **pipeline_kwargs,
        )
        model = getattr(loaded_pipeline, "model", None)
        config = getattr(model, "config", None)
        resolved_revision = getattr(config, "_commit_hash", None)
        if resolved_revision is None and len(revision) == 40:
            try:
                int(revision, 16)
            except ValueError:
                pass
            else:
                # The requested revision is already an immutable Hub commit.
                resolved_revision = revision
        precision = str(model.dtype) if getattr(model, "dtype", None) is not None else None

        return cls(
            loaded_pipeline,
            model_id=model_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            device=str(resolved_device),
            precision=precision,
            transformers_version=_package_version("transformers"),
            torch_version=str(torch.__version__),
            generation=generation,
            torch_module=torch,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def generate_answer(self, image: ImageInput, question: str) -> str:
        """Return the generated answer for one local image and question."""

        _validate_question(question)
        rgb_image = _load_rgb_image(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        try:
            if self._torch is not None:
                self._torch.manual_seed(self._generation.seed)
            output = self._pipeline(
                text=messages,
                max_new_tokens=self._generation.max_new_tokens,
                generate_kwargs={
                    "do_sample": False,
                    "num_beams": self._generation.num_beams,
                },
                return_full_text=False,
            )
        finally:
            rgb_image.close()
        return _extract_text(output)

    def generate_sampled_answers(
        self,
        image: ImageInput,
        question: str,
        settings: SamplingSettings,
    ) -> list[str]:
        """Generate repeated stochastic answers with a distinct recorded seed each time.

        This separate method leaves :meth:`generate_answer` on its existing
        deterministic greedy path. A failing generation aborts the call and
        reports its index; partial answer lists are never returned.
        """

        _validate_question(question)
        if not isinstance(settings, SamplingSettings):
            raise TypeError("settings must be a SamplingSettings instance")
        rgb_image = _load_rgb_image(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        answers: list[str] = []
        try:
            for index, seed in enumerate(settings.generation_seeds, start=1):
                try:
                    if self._torch is not None:
                        self._torch.manual_seed(seed)
                    output = self._pipeline(
                        text=messages,
                        max_new_tokens=settings.max_new_tokens,
                        generate_kwargs={
                            "do_sample": True,
                            "num_beams": 1,
                            "temperature": float(settings.temperature),
                            "top_p": float(settings.top_p),
                        },
                        return_full_text=False,
                    )
                    answers.append(_extract_text(output))
                except Exception as exc:
                    raise ModelAdapterError(
                        f"sample generation {index}/{settings.num_generations} "
                        f"(seed {seed}) failed: {type(exc).__name__}: {exc}"
                    ) from exc
        finally:
            rgb_image.close()
        return answers

    @property
    def vision_encoder_metadata(self) -> dict[str, Any]:
        """Describe the frozen visual tower and feature extraction procedure."""

        model = getattr(self._pipeline, "model", None)
        processor = getattr(self._pipeline, "processor", None)
        if model is None or processor is None:
            raise ModelAdapterError("image feature extraction requires a model and processor")
        config = getattr(model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        inner_model = getattr(model, "model", model)
        vision_tower = getattr(inner_model, "vision_model", None)
        connector = getattr(inner_model, "connector", None)
        text_config = getattr(config, "text_config", None)
        return {
            "model_identifier": self._metadata.model_id,
            "model_revision": self._metadata.resolved_revision
            or self._metadata.requested_revision,
            "model_class": type(model).__name__,
            "vision_tower_class": type(vision_tower).__name__ if vision_tower else None,
            "vision_model_type": getattr(vision_config, "model_type", None),
            "vision_hidden_size": getattr(vision_config, "hidden_size", None),
            "connector_class": type(connector).__name__ if connector else None,
            "feature_dimension": getattr(text_config, "hidden_size", None),
            "processor_class": type(processor).__name__,
            "feature_source": "model.get_image_features (frozen vision tower and connector)",
            "pooling": "arithmetic mean over projected image-token features and image crops",
            "device": self._metadata.device,
            "precision": self._metadata.precision,
            "transformers_version": self._metadata.transformers_version,
            "torch_version": self._metadata.torch_version,
        }

    def extract_image_embedding(self, image: ImageInput) -> np.ndarray:
        """Return a mean-pooled SmolVLM visual feature without running its decoder."""

        model = getattr(self._pipeline, "model", None)
        processor = getattr(self._pipeline, "processor", None)
        if model is None or processor is None:
            raise ModelAdapterError("image feature extraction requires a model and processor")
        get_image_features = getattr(model, "get_image_features", None)
        if not callable(get_image_features):
            raise ModelAdapterError("the loaded model does not expose get_image_features")
        torch = self._torch
        if torch is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise ModelAdapterError("image feature extraction requires PyTorch") from exc
            torch = torch_module

        rgb_image = _load_rgb_image(image)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": rgb_image},
                        {"type": "text", "text": "Extract image features."},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = getattr(model, "device", self._metadata.device)
            inputs = inputs.to(device)
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                raise ModelAdapterError("the processor returned no pixel_values for the image")
            with torch.inference_mode():
                features = get_image_features(
                    pixel_values=pixel_values,
                    pixel_attention_mask=inputs.get("pixel_attention_mask"),
                )
            pooled_features = getattr(features, "pooler_output", None)
            if (
                pooled_features is None
                and isinstance(features, (tuple, list))
                and len(features) > 1
            ):
                pooled_features = features[1]
            if isinstance(pooled_features, (tuple, list)):
                if len(pooled_features) != 1:
                    raise ModelAdapterError(
                        "expected one image in the feature output, "
                        f"received {len(pooled_features)} feature groups"
                    )
                pooled_features = pooled_features[0]
            if not isinstance(pooled_features, torch.Tensor):
                raise ModelAdapterError("the model returned no tensor image features")
            if pooled_features.ndim == 3:
                if pooled_features.shape[0] != 1:
                    raise ModelAdapterError("expected a single image in the feature batch")
                pooled_features = pooled_features[0]
            if pooled_features.ndim != 2 or pooled_features.shape[0] == 0:
                raise ModelAdapterError(
                    "image features must be a non-empty matrix of token features"
                )
            embedding = (
                pooled_features.float()
                .mean(dim=0)
                .detach()
                .to(device="cpu")
                .numpy()
                .astype(np.float32, copy=True)
            )
            if not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
                raise ModelAdapterError("the model returned an invalid or zero image embedding")
            return embedding
        finally:
            rgb_image.close()

    def score_candidate_answers(
        self,
        image: ImageInput,
        question: str,
        candidate_answers: tuple[str, ...] = ("Yes", "No"),
    ) -> dict[str, CandidateSequenceLogProbability]:
        """Score candidate strings with conditional sequence log probabilities.

        Each candidate is appended as an assistant turn using the loaded
        processor's chat template. The model's causal logits at each answer
        token position are normalized over the full vocabulary; the selected
        token log probabilities are summed. Trailing chat end markers are not
        included in the textual candidate likelihood.
        """

        _validate_question(question)
        if not candidate_answers or any(
            not isinstance(answer, str) or not answer.strip() for answer in candidate_answers
        ):
            raise ValueError("candidate_answers must contain non-empty strings")
        if len(set(candidate_answers)) != len(candidate_answers):
            raise ValueError("candidate_answers must be unique")

        processor = getattr(self._pipeline, "processor", None)
        model = getattr(self._pipeline, "model", None)
        if processor is None or model is None:
            raise ModelAdapterError(
                "candidate likelihood scoring requires a loaded Transformers model and processor"
            )

        torch = self._torch
        if torch is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise ModelAdapterError("candidate likelihood scoring requires PyTorch") from exc
            torch = torch_module

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ModelAdapterError("the loaded multimodal processor has no tokenizer")
        special_token_ids = set(getattr(tokenizer, "all_special_ids", ()))
        device = getattr(model, "device", self._metadata.device)
        model_dtype = getattr(model, "dtype", None)
        rgb_image = _load_rgb_image(image)

        def make_user_turn() -> dict[str, Any]:
            return {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": question},
                ],
            }

        def encode(messages: list[dict[str, Any]]) -> Any:
            encoded = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if model_dtype is not None:
                encoded = encoded.to(dtype=model_dtype)
            return encoded.to(device)

        try:
            prompt_inputs = processor.apply_chat_template(
                [make_user_turn()],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if model_dtype is not None:
                prompt_inputs = prompt_inputs.to(dtype=model_dtype)
            prompt_inputs = prompt_inputs.to(device)
            prompt_ids = prompt_inputs["input_ids"][0]
            prompt_length = int(prompt_ids.shape[0])
            if prompt_length == 0:
                raise ModelAdapterError("processor produced an empty answer prompt")

            scored: dict[str, CandidateSequenceLogProbability] = {}
            for answer in candidate_answers:
                full_inputs = encode(
                    [
                        make_user_turn(),
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answer}],
                        },
                    ]
                )
                full_ids = full_inputs["input_ids"][0]
                if full_ids.shape[0] <= prompt_length or not torch.equal(
                    full_ids[:prompt_length], prompt_ids
                ):
                    raise ModelAdapterError(
                        "candidate chat tokens do not share the generation prompt prefix; "
                        "cannot align answer-position logits safely"
                    )

                candidate_token_ids = full_ids[prompt_length:].tolist()
                terminal_marker = next(
                    (
                        index
                        for index, token_id in enumerate(candidate_token_ids)
                        if token_id in special_token_ids
                    ),
                    len(candidate_token_ids),
                )
                candidate_token_ids = candidate_token_ids[:terminal_marker]
                if not candidate_token_ids:
                    raise ModelAdapterError(
                        f"processor produced no text tokens for candidate {answer!r}"
                    )

                with torch.inference_mode():
                    outputs = model(**full_inputs)
                    logits = outputs.logits[0]
                    start = prompt_length - 1
                    stop = start + len(candidate_token_ids)
                    if stop > logits.shape[0]:
                        raise ModelAdapterError(
                            "model returned too few logits to score the candidate answer"
                        )
                    answer_step_logits = logits[start:stop].float()
                    target_ids = torch.tensor(
                        candidate_token_ids,
                        dtype=torch.long,
                        device=answer_step_logits.device,
                    )
                    token_log_probabilities = torch.log_softmax(
                        answer_step_logits,
                        dim=-1,
                    ).gather(-1, target_ids.unsqueeze(-1))
                    sequence_log_probability = float(token_log_probabilities.sum().item())

                scored[answer] = CandidateSequenceLogProbability(
                    answer_text=answer,
                    token_ids=tuple(int(token_id) for token_id in candidate_token_ids),
                    log_probability=sequence_log_probability,
                )
            return scored
        finally:
            rgb_image.close()


class MockVLMAdapter:
    """Small deterministic adapter for tests that must not load model weights."""

    def __init__(
        self,
        answer: str = "MOCK_ANSWER",
        *,
        sampled_answers: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        if not isinstance(answer, str):
            raise TypeError("mock answer must be a string")
        if sampled_answers is not None and any(
            not isinstance(sampled_answer, str) for sampled_answer in sampled_answers
        ):
            raise TypeError("sampled_answers must contain only strings")
        self._answer = answer
        self._sampled_answers = (
            tuple(sampled_answers) if sampled_answers is not None else None
        )
        self._sampled_answer_offset = 0
        self.calls: list[dict[str, str]] = []
        self.sampled_calls: list[dict[str, Any]] = []
        self._metadata = ModelMetadata(
            model_id="halluood/mock-vlm",
            requested_revision="test-only",
            resolved_revision="test-only",
            backend="mock",
            device="cpu",
            precision=None,
            transformers_version=None,
            torch_version=None,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def generate_answer(self, image: ImageInput, question: str) -> str:
        _validate_question(question)
        image_description = str(image) if isinstance(image, (str, Path)) else "<PIL.Image>"
        self.calls.append({"image": image_description, "question": question})
        return self._answer

    def generate_sampled_answers(
        self,
        image: ImageInput,
        question: str,
        settings: SamplingSettings,
    ) -> list[str]:
        """Return scripted or repeated mock answers without model inference."""

        _validate_question(question)
        if not isinstance(settings, SamplingSettings):
            raise TypeError("settings must be a SamplingSettings instance")
        image_description = str(image) if isinstance(image, (str, Path)) else "<PIL.Image>"
        self.sampled_calls.append(
            {
                "image": image_description,
                "question": question,
                "sampling_settings": settings.to_dict(),
            }
        )
        if self._sampled_answers is None:
            return [self._answer] * settings.num_generations
        start = self._sampled_answer_offset
        stop = start + settings.num_generations
        if stop > len(self._sampled_answers):
            raise ModelAdapterError(
                f"mock has {len(self._sampled_answers) - start} sampled answers remaining; "
                f"{settings.num_generations} requested"
            )
        self._sampled_answer_offset = stop
        return list(self._sampled_answers[start:stop])
