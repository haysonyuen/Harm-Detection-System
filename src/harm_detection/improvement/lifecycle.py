"""Recommendation lifecycle state machine (PRD §12.6).

proposed -> accepted_for_experiment | rejected
accepted_for_experiment -> experimented | rejected
experimented -> promoted | not_promoted
"""

from __future__ import annotations

PROPOSED = "proposed"
ACCEPTED = "accepted_for_experiment"
REJECTED = "rejected"
EXPERIMENTED = "experimented"
PROMOTED = "promoted"
NOT_PROMOTED = "not_promoted"

TRANSITIONS: dict[str, set[str]] = {
    PROPOSED: {ACCEPTED, REJECTED},
    ACCEPTED: {EXPERIMENTED, REJECTED},
    EXPERIMENTED: {PROMOTED, NOT_PROMOTED},
    REJECTED: set(),
    PROMOTED: set(),
    NOT_PROMOTED: set(),
}
TERMINAL = {REJECTED, PROMOTED, NOT_PROMOTED}


class InvalidTransitionError(Exception):
    pass


def check_transition(current: str, target: str) -> None:
    if target not in TRANSITIONS.get(current, set()):
        raise InvalidTransitionError(f"cannot move recommendation from {current!r} to {target!r}")


def can_edit(current: str) -> bool:
    return current == PROPOSED
