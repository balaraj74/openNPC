"""Player pattern tracker for adaptive enemy and villain intelligence.

Tracks player behavior over time and exposes signals that the decision engine
can use to adapt tactics. This is the "strategy memory" layer described in the
project requirements.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from time import time


@dataclass(slots=True)
class PatternSnapshot:
    """A timestamped summary of observed player behavior."""

    timestamp: float
    action_counts: dict[str, int]
    dominant_action: str
    aggression_estimate: float
    predictability: float
    approach_tendency: float


class PlayerPatternTracker:
    """Maintains a rolling window of observed player actions and infers
    tactical tendencies that the enemy/villain can exploit.
    """

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self._history: deque[str] = deque(maxlen=window_size)
        self._snapshots: list[PatternSnapshot] = []

    def record(self, action: str) -> None:
        """Record one observed player action."""
        self._history.append(action)

    @property
    def total_observations(self) -> int:
        return len(self._history)

    def dominant_action(self) -> str:
        """Return the most frequent player action in the window."""
        if not self._history:
            return "unknown"
        counts = Counter(self._history)
        return counts.most_common(1)[0][0]

    def aggression_estimate(self) -> float:
        """0-1 estimate of how aggressive the player is.

        Counts attack-like actions relative to the window.
        """
        if not self._history:
            return 0.5
        aggressive = {"attack", "flank", "set_trap", "charge"}
        count = sum(1 for action in self._history if action in aggressive)
        return count / len(self._history)

    def predictability(self) -> float:
        """0-1 estimate of how repetitive the player's pattern is.

        High values mean the player keeps using the same actions.
        """
        if len(self._history) < 3:
            return 0.0
        counts = Counter(self._history)
        top_freq = counts.most_common(1)[0][1]
        return top_freq / len(self._history)

    def approach_tendency(self) -> float:
        """0-1 estimate of how often the player closes distance.

        Actions like move, attack, charge indicate closing.
        """
        if not self._history:
            return 0.5
        closing = {"move", "attack", "charge", "flank"}
        count = sum(1 for action in self._history if action in closing)
        return count / len(self._history)

    def snapshot(self) -> PatternSnapshot:
        """Take a timestamped snapshot of the current analysis."""
        counts = dict(Counter(self._history))
        snap = PatternSnapshot(
            timestamp=time(),
            action_counts=counts,
            dominant_action=self.dominant_action(),
            aggression_estimate=self.aggression_estimate(),
            predictability=self.predictability(),
            approach_tendency=self.approach_tendency(),
        )
        self._snapshots.append(snap)
        return snap

    def summary(self) -> str:
        """Human-readable summary for injection into memory or prompts."""
        if not self._history:
            return "No player pattern data available."
        dom = self.dominant_action()
        agg = self.aggression_estimate()
        pred = self.predictability()
        app = self.approach_tendency()
        lines = [
            f"Player dominant action: {dom}",
            f"Aggression estimate: {agg:.2f}",
            f"Predictability: {pred:.2f}",
            f"Approach tendency: {app:.2f}",
        ]
        if pred > 0.5:
            lines.append(f"Player is highly predictable — exploit with counter-tactics.")
        if agg > 0.6:
            lines.append("Player is aggressive — consider traps or defensive posture.")
        elif agg < 0.3:
            lines.append("Player is passive — press the attack.")
        return " | ".join(lines)

    def counter_recommendation(self) -> str:
        """Suggest a broad tactical counter based on current patterns."""
        agg = self.aggression_estimate()
        pred = self.predictability()
        app = self.approach_tendency()

        if pred > 0.6 and agg > 0.5:
            return "set_trap"
        if agg > 0.65:
            return "seek_cover"
        if app > 0.6:
            return "flank"
        if agg < 0.25:
            return "attack"
        return "patrol"

    def to_dict(self) -> dict:
        return {
            "total_observations": self.total_observations,
            "dominant_action": self.dominant_action(),
            "aggression_estimate": round(self.aggression_estimate(), 3),
            "predictability": round(self.predictability(), 3),
            "approach_tendency": round(self.approach_tendency(), 3),
            "counter_recommendation": self.counter_recommendation(),
            "recent_actions": list(self._history),
        }
