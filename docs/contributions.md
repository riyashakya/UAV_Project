# Contributions (for the introduction) — honest, no overclaim

Paste/adapt into the dissertation introduction. Consistent with the honest positioning in
[`related_work.md`](related_work.md): the methods are established; the value is integration, rigorous
evaluation, and honest analysis. **Do not describe any method here as novel.**

> **Contributions.** This dissertation does not propose a novel algorithm. The components it uses —
> auction-based task reallocation, boustrophedon coverage, SAROPS-style drift prediction,
> hazard-weighted routing, and probability-guided search — are all established, and integrated UAV
> search-and-rescue frameworks combining detection, coordination and routing already exist (e.g.
> AI-Enhanced UAV Clusters, 2026). Its contributions are instead ones of **integration, methodology,
> and rigorous, honest evaluation**:
>
> 1. a **reproducible, decoupled simulation framework** in which real, offline-measured perception
>    error is treated as a *controlled variable* over the coordination layer;
> 2. a **systematic Monte-Carlo evaluation** (mean ± 95 % CI, with ablations) of adaptive task
>    reallocation under UAV failure, including an ablation that honestly attributes the measured
>    advantage to *generic reallocation* (+21 pts survivor detection) rather than the added
>    probability-guided search (a ≈ 1.5× time-to-locate speed-up only);
> 3. an **adaptation** of maritime drift modelling to inland-flood UAV re-tasking, with hazard-aware
>    rescue routing evaluated on real OpenStreetMap networks; and
> 4. a set of **honest negative / boundary findings** — naive SAHI tiling reduced detection AP, a
>    spiral coverage pattern fails on thin sectors, and perception error and coordination failure
>    compound.
>
> The value lies in the breadth of correct application, the transparency and rigour of the evaluation,
> and its honest positioning relative to prior art — not in a new method.

## Scope / honesty guardrails (keep these true throughout the write-up)

- Every "improves / beats" claim is scoped to **a reproduced baseline in this simulator**, not the
  state of the art.
- Coordination experiments use **synthetic survivor distributions** (documented); the real detection
  cache is saturated and used only for the perception scoring and `make sim`/`make sweep`.
- The flow field is **assumed**, not estimated from imagery; drift constants are illustrative.
- RQ4 is model **validation**, not a headline contribution.
