# ADR-003: an opt-in web visualiser, scoped so it does not become the "web dashboard" non-goal

**Status:** accepted (2026-08-02) · **Supersedes/limits:** the CLAUDE.md non-goal *"Web dashboards.
Static matplotlib / folium output only."*

## Context

The project's outputs are deliberately static (figures + CSV/JSON), and a "web dashboard" is listed
as a non-goal — the rationale being that the contribution is the *offline, reproducible simulation
and evaluation*, and a live dashboard adds surface area without adding to that contribution.

A browser-based, interactive way to run a mission and watch the fleet was nonetheless requested for
demonstration and teaching. This ADR records that decision and, more importantly, the **guardrails**
that keep it from undermining the dissertation's framing.

## Decision

Add a small **opt-in demo tool** under `webapp/` — a standard-library HTTP server (`make web`) that
runs the *existing* simulation engine on request and streams the mission to a browser canvas.

It is scoped as follows, and these constraints are the point of the ADR:

1. **Not part of the evaluated pipeline.** It lives outside `src/`. No result in the dissertation is
   produced by it; the headline numbers still come from `make sweep` (offline, seeded, batch). The
   web tool only *replays* what the engine already computes.
2. **No new dependency.** Standard-library `http.server` only — `make setup` stays lean and CPU-only.
   It is not wired into `make test`/`make demo`.
3. **ADR-001 intact.** `webapp/` is coordination-side: it imports `src.sim` / `src.coordination`
   and reads the cached detections through the oracle. It never imports the perception detector.
4. **Local only.** Bound to `127.0.0.1`; it is a demo aid on the developer's machine, not a service.
5. **Determinism preserved.** Same controls → same mission (the engine is unchanged; the server just
   passes a seed and reads back the recorded trajectory).

## Consequences

- The offline, reproducible methodology and its results are unaffected; the tool is a presentation
  layer over the same engine.
- The CLAUDE.md non-goal still holds for the *core project*: no dashboard is part of the deliverable
  or the evaluation. In the write-up, describe `webapp/` as a demonstration aid, not a contribution.
- If it ever grows a database, auth, background jobs, or a build step, revisit this ADR — that would
  be the "web dashboard" the non-goal warns against.
