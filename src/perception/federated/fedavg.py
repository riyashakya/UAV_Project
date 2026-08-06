"""FedAvg weight aggregation — the mathematical core, kept pure so it can be unit-tested.

FedAvg (McMahan et al., 2017) sets the new global weights to the sample-count-weighted average of
the clients' locally-updated weights. This does exactly that and works on any dict of array-like
tensors (numpy in the tests, torch state-dicts at run time), so it needs neither torch nor a GPU.
FedProx (Li et al., 2020) uses the *same* aggregation; its difference is a proximal term added to
each client's local loss, which lives in the client, not here.
"""

from __future__ import annotations


def weighted_average(states: list[dict], weights: list[float]) -> dict:
    """Return the per-key weighted average of client state dicts.

    ``states`` are per-client weight dicts with the same keys; ``weights`` are their sample counts.
    """
    if not states:
        raise ValueError("no client states to aggregate")
    if len(states) != len(weights):
        raise ValueError("states and weights must be the same length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    keys = states[0].keys()
    return {k: sum(w * s[k] for w, s in zip(weights, states)) / total for k in keys}
