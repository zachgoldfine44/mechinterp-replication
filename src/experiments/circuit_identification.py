"""Circuit identification experiment: find components responsible for a behavior.

Generic experiment for circuit-style mechanistic interpretability papers
(IOI, induction heads, greater-than circuit, etc.). Given a clean prompt,
a corrupted prompt, and an answer/distractor pair, ranks model components
by how much each contributes to the clean-vs-corrupted logit gap.

Three methods are supported, all dispatched from the same experiment:

- ``causal_trace`` -- Meng-style denoising patching, layer x position grid.
  Works with both TransformerLens and HuggingFace backends.
- ``edge_attribution_patching`` -- gradient-based EAP node attribution.
  Ranks per-head ``hook_z`` outputs (or residual stream nodes). Requires
  TransformerLens.
- ``head_attribution`` -- ablation-based per-head attribution. Requires
  TransformerLens.

Expected params in ``ClaimConfig.params``::

    clean_prompt: str
    corrupted_prompt: str
    answer_token: str            # text form, resolved via tokenizer
    answer_token_id: int         # alternative to answer_token
    distractor_token: str | None # for the logit-difference metric
    distractor_token_id: int | None
    method: "causal_trace" | "edge_attribution_patching" | "head_attribution"
    top_k_components: int        # default 10
    metric: "logit_diff" | "prob_diff" | "kl_divergence"  # default "logit_diff"
    component_threshold: float   # default 0.0  -- count components with |score| above this

Outputs (``ExperimentResult.metrics``)::

    top_k_components: list[(component_name, score)] -- sorted by importance
    clean_metric_value: float                       -- clean run metric
    corrupted_metric_value: float                   -- corrupted run metric
    metric_recovery: float in [0, 1]                -- best single component's recovery
    n_components_above_threshold: int               -- count where |score| > threshold
    method: str                                      -- which method ran
    metric_name: str                                 -- which metric was used

Saves intermediate output to ``results_dir/{method}_result.json`` so a
killed run can resume without recomputing the (expensive) sweep.

Usage::

    from src.experiments.circuit_identification import CircuitIdentificationExperiment
    exp = CircuitIdentificationExperiment(claim_config, "gpt2_small", data_root)
    result = exp.load_or_run(model, tokenizer, activations_cache=None)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import Tensor

from src.core.claim import ClaimConfig, ExperimentResult
from src.core.experiment import Experiment

logger = logging.getLogger(__name__)


VALID_METHODS = ("causal_trace", "edge_attribution_patching", "head_attribution")
VALID_METRICS = ("logit_diff", "prob_diff", "kl_divergence")


class CircuitIdentificationExperiment(Experiment):
    """Find the circuit (set of components) responsible for a behavior.

    The experiment runs one of three methods on a (clean, corrupted) prompt
    pair, ranks every component by its contribution to the answer-vs-
    distractor logit gap, and reports the top-k along with how much of the
    gap a single best component recovers.

    The pass/fail criterion (``evaluate``) checks whether ``metric_recovery``
    is at least ``success_threshold`` (default 0.5 from ``paper_config.yaml``).
    A recovery of >= 0.5 means a single component accounts for at least half
    of the clean-vs-corrupted gap -- a meaningful "there is a clean circuit"
    signal.
    """

    def __init__(self, config: ClaimConfig, model_key: str, data_root: Path) -> None:
        super().__init__(config, model_key, data_root)

        params = config.params

        # Required prompts
        if "clean_prompt" not in params or "corrupted_prompt" not in params:
            raise ValueError(
                "circuit_identification requires both 'clean_prompt' and "
                "'corrupted_prompt' in params."
            )
        self.clean_prompt: str = params["clean_prompt"]
        self.corrupted_prompt: str = params["corrupted_prompt"]

        # Answer / distractor (text form OR token id form)
        self.answer_token: str | None = params.get("answer_token")
        self.answer_token_id: int | None = params.get("answer_token_id")
        if self.answer_token is None and self.answer_token_id is None:
            raise ValueError(
                "circuit_identification requires either 'answer_token' or "
                "'answer_token_id' in params."
            )

        self.distractor_token: str | None = params.get("distractor_token")
        self.distractor_token_id: int | None = params.get("distractor_token_id")

        # Method selection
        method = params.get("method", "causal_trace")
        if method not in VALID_METHODS:
            raise ValueError(
                f"Unknown method: {method!r}. "
                f"Expected one of {VALID_METHODS}."
            )
        self.method: str = method

        # Metric selection
        metric = params.get("metric", "logit_diff")
        if metric not in VALID_METRICS:
            raise ValueError(
                f"Unknown metric: {metric!r}. "
                f"Expected one of {VALID_METRICS}."
            )
        self.metric: str = metric

        self.top_k: int = int(params.get("top_k_components", 10))
        self.component_threshold: float = float(params.get("component_threshold", 0.0))

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self, model: Any, tokenizer: Any, activations_cache: Any
    ) -> ExperimentResult:
        """Execute the chosen circuit-identification method and rank components.

        Args:
            model: TransformerLens HookedTransformer (preferred) or
                HuggingFace AutoModelForCausalLM. Some methods are TL-only.
            tokenizer: Paired tokenizer (used to resolve answer/distractor
                token text into vocab IDs).
            activations_cache: Unused; circuits don't share an activations
                cache the way probe experiments do.

        Returns:
            ExperimentResult with top_k_components, metric_recovery, etc.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Resume from intermediate result if available.
        intermediate = self.results_dir / f"{self.method}_result.json"
        if intermediate.exists():
            logger.info("Loading cached %s result from %s", self.method, intermediate)
            with open(intermediate) as f:
                cached = json.load(f)
            return self._build_result(cached)

        # Tokenize prompts and resolve answer/distractor IDs.
        clean_tokens, corrupted_tokens = self._tokenize_prompts(model, tokenizer)
        answer_id = self._resolve_token_id(
            self.answer_token, self.answer_token_id, tokenizer, role="answer"
        )
        distractor_id: int | None = None
        if self.distractor_token is not None or self.distractor_token_id is not None:
            distractor_id = self._resolve_token_id(
                self.distractor_token,
                self.distractor_token_id,
                tokenizer,
                role="distractor",
            )

        # Build the metric function used by EAP / head attribution. The
        # natural choice is logit difference between answer and distractor;
        # if no distractor is supplied we fall back to the answer logit alone.
        metric_fn = self._build_metric_fn(answer_id, distractor_id)

        # Dispatch.
        if self.method == "causal_trace":
            payload = self._run_causal_trace(
                model, clean_tokens, corrupted_tokens, answer_id
            )
        elif self.method == "edge_attribution_patching":
            payload = self._run_edge_attribution(
                model, clean_tokens, corrupted_tokens, metric_fn
            )
        elif self.method == "head_attribution":
            payload = self._run_head_attribution(
                model, clean_tokens, metric_fn
            )
        else:
            raise ValueError(f"Unreachable: method={self.method!r}")

        # Persist intermediate result for resume safety.
        tmp = intermediate.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        tmp.rename(intermediate)
        logger.info("Saved %s result to %s", self.method, intermediate)

        return self._build_result(payload)

    def evaluate(self, result: ExperimentResult) -> bool:
        """Pass if a single component recovers at least the threshold fraction of the gap.

        The default threshold (set in paper_config.yaml) is 0.5 -- a single
        best component accounts for >= 50% of the clean-vs-corrupted metric
        gap. This is the canonical "is there a clean, single-component
        circuit?" signal.

        ``success_metric`` should typically be ``"metric_recovery"``.
        """
        metric_val = result.metrics.get(self.config.success_metric, 0.0)
        try:
            return float(metric_val) >= float(self.config.success_threshold)
        except (TypeError, ValueError):
            return False

    # ── Private helpers: tokenization & metric ─────────────────────────────

    def _tokenize_prompts(
        self, model: Any, tokenizer: Any
    ) -> tuple[Tensor, Tensor]:
        """Tokenize clean and corrupted prompts and verify they have equal length.

        Causal tracing and path patching require the two prompts to have the
        same number of tokens so that ``(layer, position)`` indices line up.
        We raise a clear error early rather than letting the technique
        modules fail with a less informative ``shape mismatch`` later on.
        """
        # Prefer the TransformerLens helper if available.
        if hasattr(model, "to_tokens"):
            clean_tokens = model.to_tokens(self.clean_prompt)
            corrupted_tokens = model.to_tokens(self.corrupted_prompt)
        else:
            clean_tokens = tokenizer(self.clean_prompt, return_tensors="pt")["input_ids"]
            corrupted_tokens = tokenizer(self.corrupted_prompt, return_tensors="pt")["input_ids"]

        if tuple(clean_tokens.shape) != tuple(corrupted_tokens.shape):
            raise ValueError(
                f"clean and corrupted prompts must tokenize to the same length "
                f"for circuit identification. Got clean={tuple(clean_tokens.shape)} "
                f"vs corrupted={tuple(corrupted_tokens.shape)}. "
                f"Pad or rephrase the prompts so the only difference is the swapped "
                f"semantic content (this is the standard IOI / circuit-discovery setup)."
            )

        return clean_tokens, corrupted_tokens

    def _resolve_token_id(
        self,
        token_text: str | None,
        token_id: int | None,
        tokenizer: Any,
        role: str,
    ) -> int:
        """Resolve a (token_text, token_id) pair into a single vocab id.

        Prefers the explicit ``token_id`` if both are supplied. When
        resolving from text, leading whitespace matters for BPE tokenizers
        (e.g. " Mary" tokenizes to a single token but "Mary" may not). We
        try the raw text first; if it produces multiple tokens we try a
        leading-space variant.
        """
        if token_id is not None:
            return int(token_id)

        if token_text is None:
            raise ValueError(f"No {role} token specified")

        # First try: raw text.
        encoded = tokenizer(token_text, add_special_tokens=False)["input_ids"]
        if len(encoded) == 1:
            return int(encoded[0])

        # Try with a leading space (common BPE case).
        encoded_sp = tokenizer(" " + token_text, add_special_tokens=False)["input_ids"]
        if len(encoded_sp) == 1:
            return int(encoded_sp[0])

        raise ValueError(
            f"{role} token {token_text!r} did not encode to a single token "
            f"(got {encoded!r}). Pass {role}_token_id explicitly instead."
        )

    def _build_metric_fn(
        self, answer_id: int, distractor_id: int | None
    ) -> Callable[[Tensor], Tensor]:
        """Construct the logit-based metric used by EAP / head attribution.

        Returns a callable ``logits -> scalar Tensor`` so it can be used by
        ``edge_attribution_patching`` (which calls ``.backward()``) and by
        ``compute_head_attribution`` (which only needs a scalar). When a
        distractor token is supplied we use the logit difference; otherwise
        we use just the answer logit.
        """

        def metric_fn(logits: Tensor) -> Tensor:
            # logits: (batch, seq_len, vocab) -> scalar at last position.
            last = logits[0, -1]
            answer = last[answer_id]
            if distractor_id is None:
                return answer
            return answer - last[distractor_id]

        return metric_fn

    # ── Private helpers: dispatch into technique modules ──────────────────

    def _run_causal_trace(
        self,
        model: Any,
        clean_tokens: Tensor,
        corrupted_tokens: Tensor,
        answer_id: int,
    ) -> dict[str, Any]:
        """Run a denoising causal-trace sweep over (layer, position) pairs.

        Returns a JSON-serializable payload describing the result.
        """
        # Lazy import so the experiment module loads without numpy/torch
        # heavy techniques being importable in unit tests that mock everything.
        from src.techniques import patching as patching_mod

        out = patching_mod.causal_trace(
            model=model,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            answer_token_id=answer_id,
            metric=self.metric,
        )

        # Flatten effect_grid into ranked components.
        effect_grid = out["effect_grid"]  # numpy (n_layers, n_positions)
        layers = out["layers"]
        positions = out["positions"]

        ranked: list[tuple[str, float]] = []
        for li, layer in enumerate(layers):
            for pi, pos in enumerate(positions):
                name = f"L{layer}P{pos}"
                ranked.append((name, float(effect_grid[li, pi])))

        ranked.sort(key=lambda kv: abs(kv[1]), reverse=True)
        top_k = ranked[: self.top_k]
        n_above = sum(1 for _, s in ranked if abs(s) > self.component_threshold)

        clean_logit = float(out["clean_logit"])
        corrupted_logit = float(out["corrupted_logit"])

        # For causal_trace, the effect_grid is *already* the normalized
        # restoration ratio (clamped at 1.0 when the patch fully restores).
        # The best single component's recovery is just the largest entry.
        best_recovery = float(max((abs(s) for _, s in ranked), default=0.0))
        # Clamp into [0, 1] for the public metric_recovery field; values
        # >1 are possible if a single layer overshoots, but for the
        # success criterion we cap at 1.
        metric_recovery = min(1.0, max(0.0, best_recovery))

        return {
            "method": "causal_trace",
            "metric_name": self.metric,
            "top_k_components": top_k,
            "all_ranked_components": ranked,
            "clean_metric_value": clean_logit,
            "corrupted_metric_value": corrupted_logit,
            "metric_recovery": metric_recovery,
            "n_components_above_threshold": n_above,
            "best_layer": int(out["best_layer"]),
            "best_position": int(out["best_position"]),
        }

    def _run_edge_attribution(
        self,
        model: Any,
        clean_tokens: Tensor,
        corrupted_tokens: Tensor,
        metric_fn: Callable[[Tensor], Tensor],
    ) -> dict[str, Any]:
        """Run gradient-based edge attribution patching."""
        from src.techniques import circuit_discovery as cd_mod

        eap_result = cd_mod.edge_attribution_patching(
            model=model,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            metric_fn=metric_fn,
            top_k=max(self.top_k, 50),
            nodes="heads",
        )

        clean_metric = float(eap_result.metric_value_clean)
        corrupted_metric = float(eap_result.metric_value_corrupted)
        gap = clean_metric - corrupted_metric

        ranked = sorted(
            eap_result.edge_scores.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        top_k = ranked[: self.top_k]
        n_above = sum(1 for _, s in ranked if abs(s) > self.component_threshold)

        # Recovery: the top-1 first-order attribution as a fraction of the
        # clean-vs-corrupted gap. EAP scores are first-order Taylor
        # approximations, so |top| / |gap| roughly tells us how much of the
        # gap a single node "explains".
        if abs(gap) > 1e-10 and ranked:
            best_score = abs(ranked[0][1])
            metric_recovery = min(1.0, best_score / abs(gap))
        else:
            metric_recovery = 0.0

        return {
            "method": "edge_attribution_patching",
            "metric_name": self.metric,
            "top_k_components": top_k,
            "all_ranked_components": ranked,
            "clean_metric_value": clean_metric,
            "corrupted_metric_value": corrupted_metric,
            "metric_recovery": float(metric_recovery),
            "n_components_above_threshold": n_above,
            "n_edges_scored": eap_result.n_edges_scored,
        }

    def _run_head_attribution(
        self,
        model: Any,
        clean_tokens: Tensor,
        metric_fn: Callable[[Tensor], Tensor],
    ) -> dict[str, Any]:
        """Run ablation-based per-head attribution."""
        from src.techniques import attention as attn_mod

        attribution = attn_mod.compute_head_attribution(
            model=model,
            tokens=clean_tokens,
            metric_fn=metric_fn,
            ablation="zero",
        )

        # Compute the unablated baseline metric on clean_tokens.
        with torch.no_grad():
            baseline_logits = model(clean_tokens)
        clean_metric = float(metric_fn(baseline_logits).detach().item())

        # Top-k by attribution magnitude.
        ranked = sorted(
            (
                (f"L{L}H{H}", float(score))
                for (L, H), score in attribution.items()
            ),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        top_k = ranked[: self.top_k]
        n_above = sum(1 for _, s in ranked if abs(s) > self.component_threshold)

        # For head attribution we don't have an explicit "corrupted" baseline,
        # so the recovery metric is the top-1 head's effect as a fraction of
        # the clean baseline magnitude. This is a softer signal than the
        # patching-based recovery but it's the right shape for the
        # success_criterion.
        if abs(clean_metric) > 1e-10 and ranked:
            best_score = abs(ranked[0][1])
            metric_recovery = min(1.0, best_score / abs(clean_metric))
        else:
            metric_recovery = 0.0

        return {
            "method": "head_attribution",
            "metric_name": self.metric,
            "top_k_components": top_k,
            "all_ranked_components": ranked,
            "clean_metric_value": clean_metric,
            "corrupted_metric_value": 0.0,
            "metric_recovery": float(metric_recovery),
            "n_components_above_threshold": n_above,
        }

    # ── Build ExperimentResult from a payload (used both fresh + cached) ──

    def _build_result(self, payload: dict[str, Any]) -> ExperimentResult:
        """Wrap a method-specific payload into an ExperimentResult."""
        metrics: dict[str, Any] = {
            "top_k_components": payload["top_k_components"],
            "clean_metric_value": payload["clean_metric_value"],
            "corrupted_metric_value": payload["corrupted_metric_value"],
            "metric_recovery": payload["metric_recovery"],
            "n_components_above_threshold": payload["n_components_above_threshold"],
            "method": payload["method"],
            "metric_name": payload["metric_name"],
        }

        metadata: dict[str, Any] = {
            "clean_prompt": self.clean_prompt,
            "corrupted_prompt": self.corrupted_prompt,
            "method": self.method,
            "metric": self.metric,
            "top_k": self.top_k,
            "component_threshold": self.component_threshold,
        }

        return ExperimentResult(
            claim_id=self.config.claim_id,
            model_key=self.model_key,
            paper_id=self.config.paper_id,
            metrics=metrics,
            metadata=metadata,
        )
