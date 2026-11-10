from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from hallucination_detection.utils import stable_int

LETTERS = "ABCD"


@dataclass
class InferenceResult:
    records: pd.DataFrame
    features: dict[int, np.ndarray]
    model_metadata: dict[str, Any]


def parse_answer(response: str, choices: Sequence[str]) -> tuple[int | None, str]:
    """Parse an A-D answer without treating an arbitrary article 'A' as an answer."""
    text = str(response or "").strip()
    if not text:
        return None, "empty"

    patterns = (
        r"^\s*[\(\[]?([ABCD])(?:[\)\]\.:,;!-]+(?:\s|$)|\s*$)",
        r"\b(?:answer|option|choice)\s*(?:is|=|:)?\s*[\(\[]?([ABCD])\b",
        r"\b(?:correct|best)\s+(?:answer|option|choice)\s*(?:is|=|:)?\s*([ABCD])\b",
    )
    upper = text.upper()
    for pattern in patterns:
        match = re.search(pattern, upper, flags=re.IGNORECASE)
        if match:
            return LETTERS.index(match.group(1).upper()), "letter"

    normalized = _normalize_text(text)
    for index, choice in enumerate(choices):
        normalized_choice = _normalize_text(str(choice))
        if normalized == normalized_choice:
            return index, "exact_choice_text"
        if normalized in {f"the answer is {normalized_choice}", f"answer {normalized_choice}"}:
            return index, "choice_text"
    return None, "unparseable"


def resolve_layer_indices(layer_specs: Sequence[int | float], num_layers: int) -> list[int]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive.")
    resolved: list[int] = []
    for spec in layer_specs:
        if isinstance(spec, float):
            if not 0 < spec <= 1:
                raise ValueError(f"Fractional layer {spec} must be in (0, 1].")
            layer = max(1, min(num_layers, int(round(spec * num_layers))))
        else:
            layer = int(spec)
            if layer < 0:
                layer = num_layers + layer + 1
            if not 1 <= layer <= num_layers:
                raise ValueError(f"Layer {spec} resolves outside 1..{num_layers}.")
        if layer not in resolved:
            resolved.append(layer)
    return sorted(resolved)


class SyntheticInferenceRunner:
    """Deterministic offline backend used for tests and pipeline verification."""

    def __init__(self, config: Any):
        self.config = config

    def run(self, variants: pd.DataFrame) -> InferenceResult:
        seed = int(self.config.run.seed)
        layer_specs = list(self.config.features.layers)
        layers = resolve_layer_indices(layer_specs, 4)
        records: list[dict[str, Any]] = []
        layer_rows: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

        for _, row in variants.reset_index(drop=True).iterrows():
            qid = str(row["question_id"])
            variant = str(row["variant"])
            gold = int(row["answer_index"])
            base_error = stable_int(seed, qid, "base", modulo=5) in {0, 1}
            wording_flip = variant != "canonical" and stable_int(
                seed, qid, variant, "flip", modulo=11
            ) == 0
            is_error = bool(base_error) ^ bool(wording_flip)
            predicted = (gold + 1 + stable_int(qid, variant, modulo=3)) % 4 if is_error else gold
            response = LETTERS[predicted]

            records.append(
                {
                    "variant_id": str(row["variant_id"]),
                    "question_id": qid,
                    "subject": str(row["subject"]),
                    "variant": variant,
                    "semantic_status": str(row.get("semantic_status", "synthetic")),
                    "split": str(row.get("split", "unassigned")),
                    "answer_index": gold,
                    "response_text": response,
                    "parsed_answer_index": predicted,
                    "parse_valid": True,
                    "parse_reason": "synthetic",
                    "is_correct": not is_error,
                    "is_error": is_error,
                    "prompt_tokens": len(str(row.get("prompt", "")).split()),
                }
            )

            for layer in layers:
                rng = np.random.default_rng(stable_int(seed, qid, variant, layer))
                vector = rng.normal(0.0, 0.55, size=32).astype(np.float32)
                signal = 0.7 + 0.35 * layer
                vector[0] += signal if is_error else -signal
                if variant == "direct":
                    vector[1] += 2.5
                elif variant == "exam":
                    vector[1] -= 2.5
                elif variant == "minimal":
                    vector[2] += 2.5
                vector[3 + stable_int(row["subject"], modulo=4)] += 1.0
                layer_rows[layer].append(vector)

        return InferenceResult(
            records=pd.DataFrame.from_records(records),
            features={layer: np.stack(rows) for layer, rows in layer_rows.items()},
            model_metadata={
                "backend": "synthetic",
                "model_id": self.config.model.model_id,
                "revision": self.config.model.revision,
                "num_hidden_layers": 4,
                "selected_layers": layers,
            },
        )


class HuggingFaceInferenceRunner:
    """Extract prompt-only hidden states and then generate an answer."""

    def __init__(self, config: Any):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_config = self.config.model
        requested_device = str(model_config.device).lower()
        if requested_device == "auto":
            if torch.cuda.is_available():
                requested_device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                requested_device = "mps"
            else:
                requested_device = "cpu"
        self.device = torch.device(requested_device)

        requested_dtype = str(model_config.dtype).lower()
        if requested_dtype == "auto":
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if requested_dtype not in dtype_map:
                raise ValueError(f"Unsupported model dtype: {model_config.dtype}")
            dtype = dtype_map[requested_dtype]

        revision = str(model_config.revision or "main")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_id,
            revision=revision,
            trust_remote_code=False,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            revision=revision,
            torch_dtype=dtype,
            trust_remote_code=False,
        )
        self.model.to(self.device)
        self.model.eval()

    def run(self, variants: pd.DataFrame) -> InferenceResult:
        import torch

        if self.model is None:
            self._load()
        assert self.model is not None and self.tokenizer is not None and self.device is not None

        num_layers = int(self.model.config.num_hidden_layers)
        layers = resolve_layer_indices(list(self.config.features.layers), num_layers)
        batch_size = int(self.config.model.batch_size)
        records: list[dict[str, Any]] = []
        layer_rows: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

        torch.manual_seed(int(self.config.run.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.run.seed))

        frame = variants.reset_index(drop=True)
        for start in range(0, len(frame), batch_size):
            batch_rows = frame.iloc[start : start + batch_size]
            encoded = self._encode_prompts(batch_rows["prompt"].astype(str).tolist())
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            attention_mask = encoded["attention_mask"]
            positions = torch.arange(attention_mask.shape[1], device=self.device).unsqueeze(0)
            last_positions = (positions * attention_mask).max(dim=1).values

            with torch.inference_mode():
                output = self.model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                for layer in layers:
                    state = output.hidden_states[layer]
                    selected = state[
                        torch.arange(state.shape[0], device=self.device), last_positions
                    ]
                    layer_rows[layer].extend(selected.float().cpu().numpy())

                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=int(self.config.model.max_new_tokens),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            responses = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            token_counts = attention_mask.sum(dim=1).cpu().tolist()

            for (_, row), response, token_count in zip(
                batch_rows.iterrows(), responses, token_counts, strict=True
            ):
                choices = row["choices"]
                parsed, reason = parse_answer(response, choices)
                gold = int(row["answer_index"])
                valid = parsed is not None
                correct = bool(valid and parsed == gold)
                records.append(
                    {
                        "variant_id": str(row["variant_id"]),
                        "question_id": str(row["question_id"]),
                        "subject": str(row["subject"]),
                        "variant": str(row["variant"]),
                        "semantic_status": str(row.get("semantic_status", "unknown")),
                        "split": str(row.get("split", "unassigned")),
                        "answer_index": gold,
                        "response_text": response.strip(),
                        "parsed_answer_index": parsed,
                        "parse_valid": valid,
                        "parse_reason": reason,
                        "is_correct": correct,
                        "is_error": not correct,
                        "prompt_tokens": int(token_count),
                    }
                )
            del output, generated, encoded

        metadata = {
            "backend": "huggingface",
            "model_id": self.config.model.model_id,
            "requested_revision": self.config.model.revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "num_hidden_layers": num_layers,
            "selected_layers": layers,
            "pooling": "last_prompt_token_at_assistant_boundary",
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": int(self.config.model.max_new_tokens),
            },
        }
        return InferenceResult(
            records=pd.DataFrame.from_records(records),
            features={layer: np.stack(rows).astype(np.float32) for layer, rows in layer_rows.items()},
            model_metadata=metadata,
        )

    def _encode_prompts(self, prompts: list[str]) -> dict[str, Any]:
        assert self.tokenizer is not None
        encoded_rows: list[dict[str, list[int]]] = []
        max_length = int(self.config.model.max_input_tokens)
        system_prompt = "Answer the multiple-choice question using only A, B, C, or D."
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            if getattr(self.tokenizer, "chat_template", None):
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                )
            else:
                encoded = self.tokenizer(
                    f"{system_prompt}\n\n{prompt}\n\nAnswer:",
                    add_special_tokens=True,
                )
            ids = list(encoded["input_ids"])[-max_length:]
            mask = list(encoded.get("attention_mask", [1] * len(ids)))[-max_length:]
            encoded_rows.append({"input_ids": ids, "attention_mask": mask})
        return self.tokenizer.pad(encoded_rows, padding=True, return_tensors="pt")

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def make_inference_runner(config: Any) -> SyntheticInferenceRunner | HuggingFaceInferenceRunner:
    backend = str(config.model.backend).lower()
    if backend == "synthetic":
        return SyntheticInferenceRunner(config)
    if backend == "huggingface":
        return HuggingFaceInferenceRunner(config)
    raise ValueError(f"Unsupported inference backend: {config.model.backend}")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()

