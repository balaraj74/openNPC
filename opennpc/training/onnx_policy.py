"""ONNX-based inference policy for OpenNPC.

Allows trained PPO/DQN models to be exported to ONNX format and then served
at runtime without a PyTorch dependency, using only ``onnxruntime``.

Typical workflow
----------------
1. Train a policy in Python with PyTorch::

       from opennpc.training.ppo import PPOPytorchPolicy
       trainer = PPOPytorchPolicy(state_size=8, action_space=["idle", "attack"])
       trainer.train(env, episodes=500)
       trainer.export_onnx("enemy_policy.onnx")

2. Load it at runtime::

       from opennpc.training.onnx_policy import ONNXPolicy
       policy = ONNXPolicy.from_file("enemy_policy.onnx")

3. Register it in the engine::

       engine.register_policy("enemy_v1", policy)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from opennpc.policy import Policy, PolicyDecision
from opennpc.types import AgentConfig, GameState

_ONNXRUNTIME_AVAILABLE = False
try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _ONNXRUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    ort = None  # type: ignore[assignment]


def _require_onnxruntime() -> None:
    if not _ONNXRUNTIME_AVAILABLE:
        raise ImportError(
            "onnxruntime is required for ONNXPolicy. "
            "Install it with: pip install onnxruntime"
        )


def _build_state_vector(state: GameState, state_size: int) -> np.ndarray:
    """Convert a GameState into a flat float32 vector of fixed length."""
    raw = [
        float(state.health) / 100.0,
        float(state.threat_level),
        float(state.distance_to_target or 0.0) / 50.0,
        float(state.target_health or 100.0) / 100.0,
        1.0 if state.cover_available else 0.0,
        1.0 if state.inventory else 0.0,
        float(state.tick % 100) / 100.0,
        float(len(state.nearby_entities)) / 10.0,
    ]
    raw = raw[:state_size]
    if len(raw) < state_size:
        raw.extend([0.0] * (state_size - len(raw)))
    return np.array(raw, dtype=np.float32)


class ONNXPolicy(Policy):
    """Runtime policy backed by an ONNX model.

    The model must accept a single input of shape ``(1, state_size)`` and
    return logits of shape ``(1, num_actions)``.
    """

    name = "onnx"

    def __init__(
        self,
        session: Any,
        action_space: list[str],
        state_size: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        _require_onnxruntime()
        self._session = session
        self.action_space = action_space
        self.state_size = state_size
        self._metadata = metadata or {}
        self._input_name: str = self._session.get_inputs()[0].name

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "ONNXPolicy":
        """Load an ONNX model exported by ``PPOPytorchPolicy.export_onnx()``
        or ``DQNPolicy.export_onnx()``.

        The file may be a raw ``.onnx`` file or a ``.opennpc.onnx`` bundle
        that embeds ``action_space`` and ``state_size`` as metadata.
        """
        _require_onnxruntime()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        meta_path = path.with_suffix(".onnx.meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            action_space: list[str] = meta["action_space"]
            state_size: int = int(meta["state_size"])
            extra_meta = meta
        else:
            # Fallback: read custom metadata embedded by ONNX model producer.
            sess_options = ort.SessionOptions()
            tmp_session = ort.InferenceSession(str(path), sess_options)
            props = {p.key: p.value for p in tmp_session.get_modelmeta().custom_metadata_map.items()}
            action_space_raw = props.get("action_space", "[]")
            action_space = json.loads(action_space_raw)
            state_size = int(props.get("state_size", "8"))
            extra_meta = props

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), sess_options)
        return cls(session, action_space, state_size, extra_meta)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        config: AgentConfig,
        state: GameState,
        valid_actions: list[str],
        context: dict[str, Any],
    ) -> PolicyDecision:
        state_vec = _build_state_vector(state, self.state_size)
        input_tensor = state_vec.reshape(1, -1)

        outputs = self._session.run(None, {self._input_name: input_tensor})
        logits: np.ndarray = np.array(outputs[0]).flatten()

        # Mask out actions not in valid_actions or not in our action_space.
        valid_set = set(valid_actions)
        scores: list[tuple[float, str]] = []
        for idx, action in enumerate(self.action_space):
            if action in valid_set and idx < len(logits):
                scores.append((float(logits[idx]), action))

        if not scores:
            # All masked — pick the first valid action as safe fallback.
            action = valid_actions[0] if valid_actions else "idle"
            return PolicyDecision(
                action=action,
                confidence=0.01,
                reason="ONNX: no action overlapped with valid_actions; using first valid.",
                metadata={"policy": "onnx"},
            )

        scores.sort(reverse=True)
        best_score, best_action = scores[0]

        # Softmax over valid action scores → confidence.
        raw_scores = np.array([s for s, _ in scores])
        exp_scores = np.exp(raw_scores - raw_scores.max())
        softmax_scores = exp_scores / exp_scores.sum()
        confidence = float(softmax_scores[0])

        return PolicyDecision(
            action=best_action,
            confidence=round(min(0.99, max(0.01, confidence)), 4),
            reason=f"ONNX model selected {best_action!r} (logit={best_score:.3f}).",
            metadata={"policy": "onnx", "raw_logit": best_score},
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def input_shape(self) -> tuple[int, int]:
        return (1, self.state_size)

    def __repr__(self) -> str:
        return (
            f"ONNXPolicy(state_size={self.state_size}, "
            f"actions={self.action_space})"
        )
