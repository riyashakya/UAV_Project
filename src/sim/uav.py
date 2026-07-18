"""UAV kinematics and energy model — Phase 4.

Single responsibility: a single UAV's motion (constant cruise speed; instant turns are
acceptable and documented) and energy budget ``P = P_hover + k*v**2``. Triggers
return-to-home when remaining energy < ``rth_margin * E_required_to_reach_base``; the margin
(default 1.3) lives in config, never hard-coded here.

Stub: implemented in Phase 4.
"""

from __future__ import annotations


class UAV:
    """Simulated UAV: kinematics + energy model + RTH logic. Stub (Phase 4)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("Phase 4: UAV model is not implemented yet.")
