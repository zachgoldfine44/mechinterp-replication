"""Stimulus/dataset generation and loading.

Supports four stimulus types defined in stimuli_config.yaml:
- generated: Use LLM to create stimuli (resume-aware, per-item saves)
- hardcoded: Load from JSON file
- dataset: Load from HuggingFace dataset
- programmatic: Generate from templates with variable substitution

Each loader returns a list of dicts with at least:
    {"text": str, "concept": str, "id": str}

Usage:
    from src.utils.datasets import load_stimuli

    stimuli = load_stimuli(
        stimulus_set_config={"type": "programmatic", "templates": [...]},
        paper_id="emotions",
        data_root=get_data_root(),
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_stimuli(
    stimulus_set_config: dict[str, Any],
    paper_id: str,
    data_root: Path,
    *,
    model: Any = None,
    tokenizer: Any = None,
    fast: bool = False,
) -> list[dict[str, Any]]:
    """Load or generate stimuli based on the stimulus set config.

    Routes to the correct loader based on the ``type`` field.

    Args:
        stimulus_set_config: A single entry from stimuli_config.yaml's
            ``stimulus_sets`` dict.  Must contain a ``type`` field.
        paper_id: Paper identifier (e.g., 'emotions').
        data_root: Root directory for persistent data (from get_data_root()).
        model: Optional loaded model for type="generated".
        tokenizer: Optional tokenizer for type="generated".
        fast: If True, reduce to 2 concepts and 10 stimuli per concept
            for development speed.

    Returns:
        List of dicts, each with at least {"text": str, "concept": str, "id": str}.

    Raises:
        ValueError: If the ``type`` field is unrecognised.
    """
    stim_type = stimulus_set_config.get("type", "")

    if stim_type == "generated":
        return generate_stimuli(
            stimulus_set_config, paper_id, data_root,
            model=model, tokenizer=tokenizer, fast=fast,
        )
    elif stim_type == "hardcoded":
        return load_hardcoded_stimuli(stimulus_set_config, data_root, fast=fast)
    elif stim_type == "dataset":
        return load_dataset_stimuli(stimulus_set_config, data_root, fast=fast)
    elif stim_type == "programmatic":
        return load_programmatic_stimuli(stimulus_set_config, fast=fast)
    else:
        raise ValueError(
            f"Unknown stimulus type: {stim_type!r}. "
            f"Expected one of: generated, hardcoded, dataset, programmatic"
        )


# ---------------------------------------------------------------------------
# Type: generated
# ---------------------------------------------------------------------------

def generate_stimuli(
    config: dict[str, Any],
    paper_id: str,
    data_root: Path,
    *,
    model: Any = None,
    tokenizer: Any = None,
    fast: bool = False,
) -> list[dict[str, Any]]:
    """Generate stimuli using an LLM, with resume-aware per-item saves.

    If no model is provided, falls back to creating placeholder stimuli
    for testing/development.

    Per-item save pattern: each stimulus is saved as an individual JSON file
    so that interrupted runs can resume without re-generating completed items.

    Args:
        config: Stimulus set config dict with keys:
            - generation_prompt: Template with {concept} placeholder.
            - per_concept: Number of stimuli per concept.
            - concepts: List of concept strings.
            - output_dir: Relative path under data_root for saved files.
        paper_id: Paper identifier.
        data_root: Root directory for persistent data.
        model: Loaded LLM for generation (None -> placeholders).
        tokenizer: Model's tokenizer (None -> placeholders).
        fast: If True, limit to 2 concepts and 10 stimuli per concept.

    Returns:
        List of stimulus dicts.
    """
    concepts = list(config.get("concepts", []))
    per_concept = int(config.get("per_concept", 50))
    prompt_template = config.get("generation_prompt", "")
    output_dir_rel = config.get("output_dir", f"data/{paper_id}/generated/")

    if fast:
        concepts = concepts[:2]
        per_concept = min(per_concept, 10)

    output_dir = data_root / output_dir_rel
    output_dir.mkdir(parents=True, exist_ok=True)

    # If no model available, fall back to placeholders
    if model is None or tokenizer is None:
        logger.info(
            "No model provided for generation; creating placeholders "
            "(%d concepts x %d per concept)",
            len(concepts), per_concept,
        )
        return create_placeholder_stimuli(
            concepts, per_concept, paper_id, data_root,
        )

    all_stimuli: list[dict[str, Any]] = []

    for concept in concepts:
        for idx in range(per_concept):
            stim_id = f"{concept}_{idx:04d}"
            out_path = output_dir / f"{stim_id}.json"

            # Resume: skip if already generated
            if out_path.exists():
                with open(out_path) as f:
                    all_stimuli.append(json.load(f))
                continue

            # Generate using the model
            prompt = prompt_template.replace("{concept}", concept)
            try:
                text = _generate_single(model, tokenizer, prompt)
            except Exception as exc:
                logger.warning(
                    "Generation failed for %s: %s", stim_id, exc,
                )
                text = f"[generation failed: {exc}]"

            stimulus = {
                "id": stim_id,
                "concept": concept,
                "text": text,
                "paper_id": paper_id,
                "source": "generated",
            }

            # Atomic save
            tmp_path = out_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(stimulus, f, indent=2)
            tmp_path.rename(out_path)

            all_stimuli.append(stimulus)

        logger.info(
            "Generated %d stimuli for concept '%s'", per_concept, concept,
        )

    logger.info("Total generated stimuli: %d", len(all_stimuli))
    return all_stimuli


def _generate_single(model: Any, tokenizer: Any, prompt: str) -> str:
    """Generate a single stimulus text using the model.

    Supports both TransformerLens HookedTransformer (which has .generate())
    and HuggingFace AutoModelForCausalLM.

    Args:
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        prompt: Full generation prompt.

    Returns:
        Generated text string (prompt stripped).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Move to model's device
    if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
        # TransformerLens
        device = str(model.cfg.device)
    elif hasattr(model, "device"):
        device = str(model.device)
    else:
        device = "cpu"

    input_ids = input_ids.to(device)

    with torch.no_grad():
        if hasattr(model, "generate"):
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
            )
        else:
            # Fallback: manual forward pass + greedy (shouldn't normally happen)
            logger.warning("Model has no generate(); using manual greedy decoding")
            generated = input_ids
            for _ in range(256):
                logits = model(generated)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
            output_ids = generated

    # Decode only the new tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


# ---------------------------------------------------------------------------
# Type: hardcoded
# ---------------------------------------------------------------------------

def load_hardcoded_stimuli(
    config: dict[str, Any],
    data_root: Path,
    *,
    fast: bool = False,
) -> list[dict[str, Any]]:
    """Load stimuli from a JSON file on disk.

    The JSON file should contain a list of dicts, each with at least
    'text' and 'concept' fields.  An 'id' field is added if missing.

    Args:
        config: Stimulus set config with a 'file' key (path relative to
            data_root).
        data_root: Root directory for persistent data.
        fast: If True, limit to 2 concepts and 10 per concept.

    Returns:
        List of stimulus dicts.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    file_rel = config["file"]
    file_path = data_root / file_rel

    if not file_path.exists():
        logger.warning(
            "Hardcoded stimuli file not found: %s. "
            "Returning empty list -- create this file to proceed.",
            file_path,
        )
        return []

    with open(file_path) as f:
        stimuli = json.load(f)

    if not isinstance(stimuli, list):
        raise ValueError(
            f"Expected a JSON list in {file_path}, got {type(stimuli).__name__}"
        )

    # Ensure every entry has an 'id'
    for i, stim in enumerate(stimuli):
        if "id" not in stim:
            concept = stim.get("concept", "unknown")
            stim["id"] = f"{concept}_{i:04d}"

    if fast:
        stimuli = _apply_fast_filter(stimuli)

    logger.info("Loaded %d hardcoded stimuli from %s", len(stimuli), file_path)
    return stimuli


# ---------------------------------------------------------------------------
# Type: dataset (HuggingFace)
# ---------------------------------------------------------------------------

def load_dataset_stimuli(
    config: dict[str, Any],
    data_root: Path,
    *,
    fast: bool = False,
) -> list[dict[str, Any]]:
    """Load stimuli from a HuggingFace dataset.

    Args:
        config: Stimulus set config with keys:
            - source: Must be "huggingface".
            - dataset_id: HuggingFace dataset identifier.
            - split: Dataset split (e.g., "train").
            - text_column: Column name for text (default: "text").
            - concept_column: Column name for concept/label (default: "label").
        data_root: Root directory (used for caching).
        fast: If True, limit to first 20 items.

    Returns:
        List of stimulus dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "HuggingFace datasets library not installed. "
            "Install with: pip install datasets"
        )
        return []

    dataset_id = config["dataset_id"]
    split = config.get("split", "train")
    text_col = config.get("text_column", "text")
    concept_col = config.get("concept_column", "label")

    logger.info("Loading HuggingFace dataset: %s (split=%s)", dataset_id, split)

    cache_dir = data_root / "cache" / "hf_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_id, split=split, cache_dir=str(cache_dir))

    stimuli: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        text = row.get(text_col, "")
        concept = str(row.get(concept_col, "unknown"))
        stimuli.append({
            "id": f"hf_{i:06d}",
            "text": text,
            "concept": concept,
            "source": "huggingface",
            "dataset_id": dataset_id,
        })

    if fast:
        stimuli = stimuli[:20]

    logger.info("Loaded %d stimuli from HuggingFace dataset %s", len(stimuli), dataset_id)
    return stimuli


# ---------------------------------------------------------------------------
# Type: programmatic
# ---------------------------------------------------------------------------

def load_programmatic_stimuli(
    config: dict[str, Any],
    *,
    fast: bool = False,
) -> list[dict[str, Any]]:
    """Generate stimuli from templates with variable substitution.

    For each template, substitutes each value of the specified variable
    to produce one stimulus per value.

    Args:
        config: Stimulus set config with a 'templates' key containing a list
            of template dicts, each with:
            - id: Template identifier.
            - template: String with {X} placeholder.
            - variable: Name of the variable (for metadata).
            - values: List of values to substitute.
            - expected_response: Optional dict of concept -> expected direction.
        fast: If True, limit to first 2 templates and first 3 values each.

    Returns:
        List of stimulus dicts.
    """
    templates = config.get("templates", [])

    if fast:
        templates = templates[:2]

    stimuli: list[dict[str, Any]] = []

    for tmpl in templates:
        tmpl_id = tmpl.get("id", "template")
        template_str = tmpl["template"]
        variable = tmpl.get("variable", "X")
        values = tmpl.get("values", [])
        variable_label = tmpl.get("variable_label", variable)
        expected = tmpl.get("expected_response", {})

        if fast:
            values = values[:3]

        for val in values:
            # Substitute the variable placeholder
            text = template_str.replace(f"{{{variable}}}", str(val))
            stim_id = f"{tmpl_id}_{variable_label}_{val}"

            stimulus: dict[str, Any] = {
                "id": stim_id,
                "text": text,
                "concept": "parametric",
                "source": "programmatic",
                "template_id": tmpl_id,
                variable_label: val,
                "expected_response": expected,
            }
            stimuli.append(stimulus)

    logger.info(
        "Generated %d programmatic stimuli from %d templates",
        len(stimuli), len(templates),
    )
    return stimuli


# ---------------------------------------------------------------------------
# Placeholders (for testing / development)
# ---------------------------------------------------------------------------

def create_placeholder_stimuli(
    concepts: list[str],
    n_per_concept: int,
    paper_id: str,
    data_root: Path,
) -> list[dict[str, Any]]:
    """Create simple placeholder stimuli for testing and development.

    Generates formulaic stimuli that are useful for verifying pipeline
    infrastructure without needing a real generative model.

    Args:
        concepts: List of concept strings (e.g., emotion names).
        n_per_concept: Number of placeholder stimuli per concept.
        paper_id: Paper identifier.
        data_root: Root directory for persistent data.

    Returns:
        List of placeholder stimulus dicts.
    """
    output_dir = data_root / "data" / paper_id / "placeholders"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sentence templates for variety
    templates = [
        "I am feeling very {concept} right now. This is a story about experiencing {concept}.",
        "Today has been a day full of {concept}. Everything around me reflects this {concept}.",
        "As I sit here, the feeling of {concept} washes over me completely.",
        "I woke up this morning with an overwhelming sense of {concept}.",
        "There is no other way to describe this moment except pure {concept}.",
        "The world seems different when you experience {concept} this intensely.",
        "I never knew {concept} could feel like this until today.",
        "Everyone around me can tell that I am experiencing deep {concept}.",
        "This {concept} started slowly but now it fills every part of me.",
        "If I had to capture this feeling of {concept} in words, I would say it consumes me.",
    ]

    stimuli: list[dict[str, Any]] = []

    for concept in concepts:
        for idx in range(n_per_concept):
            stim_id = f"placeholder_{concept}_{idx:04d}"
            template = templates[idx % len(templates)]
            text = template.replace("{concept}", concept)

            stimulus = {
                "id": stim_id,
                "concept": concept,
                "text": text,
                "paper_id": paper_id,
                "source": "placeholder",
            }

            # Save individual file (resume-aware pattern)
            out_path = output_dir / f"{stim_id}.json"
            if not out_path.exists():
                tmp = out_path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(stimulus, f, indent=2)
                tmp.rename(out_path)

            stimuli.append(stimulus)

    logger.info(
        "Created %d placeholder stimuli (%d concepts x %d each)",
        len(stimuli), len(concepts), n_per_concept,
    )
    return stimuli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_fast_filter(
    stimuli: list[dict[str, Any]],
    max_concepts: int = 2,
    max_per_concept: int = 10,
) -> list[dict[str, Any]]:
    """Reduce a stimulus list for --fast mode.

    Keeps at most ``max_concepts`` distinct concepts and at most
    ``max_per_concept`` stimuli per concept.

    Args:
        stimuli: Full list of stimuli.
        max_concepts: Maximum number of distinct concepts to keep.
        max_per_concept: Maximum stimuli per concept.

    Returns:
        Filtered list.
    """
    # Discover unique concepts in order of first appearance
    seen_concepts: list[str] = []
    for s in stimuli:
        c = s.get("concept", "unknown")
        if c not in seen_concepts:
            seen_concepts.append(c)

    keep_concepts = set(seen_concepts[:max_concepts])
    counts: dict[str, int] = {}
    filtered: list[dict[str, Any]] = []

    for s in stimuli:
        c = s.get("concept", "unknown")
        if c not in keep_concepts:
            continue
        counts[c] = counts.get(c, 0) + 1
        if counts[c] <= max_per_concept:
            filtered.append(s)

    return filtered
