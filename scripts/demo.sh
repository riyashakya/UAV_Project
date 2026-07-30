#!/usr/bin/env bash
# End-to-end CPU demo of the Multi-UAV disaster-response framework (no UI — CLI + static figures).
# Runs the whole story in order and drops figures in outputs/. Pauses between steps so a presenter
# can talk; run `make demo PAUSE=0` (or PAUSE=0 bash scripts/demo.sh) to go straight through.
#
# Nothing here needs a GPU or the raw imagery: perception ran offline and its detections are cached
# to data/cache/detections.parquet, which the simulator reads.

UV="${UV:-uv}"
PAUSE="${PAUSE:-1}"
SCEN="${SCEN:-configs/scenario/flood_a.yaml}"
cd "$(dirname "$0")/.." || exit 1

step() { printf '\n\033[1;36m=== %s ===\033[0m\n%s\n\n' "$1" "$2"; }
pause() { [ "$PAUSE" = "1" ] && read -rp $'\033[2m  ↵ Enter for the next step…\033[0m ' _ || true; }

printf '\n\033[1mMulti-UAV Disaster Response — live demo\033[0m\n'

step "1/6  One rescue mission" \
  "4 UAVs survey the flood area. Watch coverage=100%, the survivors/hazards found, and that every
   UAV lands safely. Same seed -> identical run (deterministic)."
$UV run python -m src.sim.engine --scenario "$SCEN" --seed 0
pause

step "2/6  Headline result — adaptive vs static under failure (1,800 runs)" \
  "The core contribution: the adaptive auction keeps far more of the area covered than a static
   plan when UAVs fail — reported with 95% confidence intervals."
$UV run python -m src.eval.runner
pause

step "3/6  Survivor-drift projection (figure)" \
  "A survivor in floodwater drifts. This projects where they are NOW (50/90% search zones) and the
   grid cells the auction would re-task toward -> outputs/drift/<ts>/drift.png"
$UV run python -m src.drift.visualize
pause

step "4/6  Drift-aware re-tasking (RQ4) — same mission, drift OFF then ON" \
  "With --drift-retask, a survivor detection re-prioritises the cells they are drifting into.
   (On this saturated scenario final coverage is unchanged — the mechanism, not a field result.)"
$UV run python -m src.sim.engine --scenario "$SCEN" --seed 0            | grep '\[sim\]'
$UV run python -m src.sim.engine --scenario "$SCEN" --seed 0 --drift-retask | grep '\[sim\]'
pause

step "5/6  Hazard-aware routing on the synthetic grid" \
  "A menu of rescue routes trading distance against risk (a Pareto front) -> outputs/routing/"
$UV run python -m src.routing.safe_path
pause

step "6/6  Hazard-aware routing on a REAL London street network" \
  "Same idea on a real cached OSM map: shortest-through-flood vs safest-detour, with a map figure."
$UV run python -m src.routing.safe_path osm

printf '\n\033[1;32mDemo complete.\033[0m Figures written under outputs/ :\n'
printf '  drift:      %s\n' "$(ls -td outputs/drift/*/ 2>/dev/null | head -1)drift.png"
printf '  routing:    %s\n' "$(ls -td outputs/routing/flood_a_*/ 2>/dev/null | head -1)pareto.png"
printf '  real map:   %s\n' "$(ls -td outputs/routing/osm_*/ 2>/dev/null | head -1)map.png"
printf '  sweep:      %s\n' "$(ls -td outputs/runs/sweep_*/ 2>/dev/null | head -1)headline.txt"
printf '\nOpen a figure (macOS), e.g.:\n  open "%smap.png"\n' "$(ls -td outputs/routing/osm_*/ 2>/dev/null | head -1)"
