# Methodology (draft chapter)

> Draft for the dissertation methodology chapter. Describes the research design, the system, and the
> experimental design. Consistent with the honest positioning in [`related_work.md`](related_work.md):
> the methods applied are established; the methodological contribution is the *decoupled, reproducible
> evaluation design*.

## 1. Research design

The study is **simulation-based, offline, and batch** — there are no physical drones, no
flight-hardware integration, and no real-time performance targets (these are explicit non-goals). The
approach is **quantitative and comparative**: each claim is evaluated by running the simulator many
times under controlled conditions and comparing an adaptive method against baselines with confidence
intervals. Perception is evaluated separately, on real labelled imagery, against ground truth.

Three design decisions are recorded as Architecture Decision Records and shape the whole methodology:

- **ADR-001 — perception is decoupled from coordination.** The simulator never runs the detector;
  perception is computed offline and cached, and the simulator reads it through one oracle interface.
- **ADR-002 — two perception models.** A detector (people, vehicles) and a segmenter (water, damaged
  buildings, blocked roads), because no single dataset covers all classes and the label types differ.
- **ADR-003 — the interactive web tool is a demonstration aid**, outside the evaluated pipeline.

## 2. Datasets and preparation

Four public datasets are unified into two training sets, plus OpenStreetMap for real road networks:

| Source | Role | Licence note |
|---|---|---|
| VisDrone (Zhu et al., 2021) | detector: small-object UAV `person`/`vehicle` | research use |
| SARD (Sambolek & Ivašić-Kos, 2021) | detector: SAR `person` | research use |
| RescueNet (Chowdhury et al., 2023) | segmenter: post-disaster damage | **CC BY-NC-ND** — not redistributed |
| FloodNet (Rahnemoonfar et al., 2021) | segmenter: flood scenes | research use |
| OpenStreetMap | real road graph for routing | © OSM contributors, ODbL |

Loaders normalise each source's native format into Ultralytics-style detect (boxes) and segment
(polygons) sets. A key, honest constraint is stated up front: **no disaster dataset labels
survivors**, so `person` is trained on ordinary/SAR imagery and transferred into the disaster domain;
this domain gap is measured (Results §2), not hidden. All datasets are cited in `docs/datasets.md`.

## 3. Perception (offline)

Two YOLO11s models are trained locally (Apple-Silicon MPS, 640 px, 60 epochs). **Model A** detects
`person`/`vehicle`; **Model B** segments `water`/`building_damaged`/`road_blocked` as masks (boxes
are not used for extents). Evaluation reports whole-image mAP and, crucially, **AP stratified by
object size** (COCO small/medium/large), because ≈ 68 % of UAV objects are < 32×32 px and whole-image
mAP alone hides the small-survivor regime that matters. Tiled inference (SAHI) is evaluated as a
candidate improvement and reported honestly, including where it does not help.

**The oracle (ADR-001 bridge).** A single offline pass caches every detection — class, confidence,
lat/lon, bounding box — keyed by `(scenario, cell)`. During a mission, when a UAV enters a cell the
oracle returns that cell's cached detections, applying a configurable **per-class false-negative
rate** and **reporting latency**, deterministic under a seed. This is the only coupling between
perception and coordination, and it is what lets detector error be treated as a *controlled variable*
(§7, RQ5).

## 4. Simulation environment

- **World.** The area of interest is a metric grid of equal cells with a base location, a per-cell
  priority field, and an analytic **flow field** `u(x,y) → (vx,vy)` (uniform / channel / radial)
  representing the flood current. Coordinates are stored in WGS84 and computed in local metres.
- **UAV.** Each UAV is a point-mass with an energy model `P = P_hover + k·v²` and a return-to-home
  rule triggered at a safe multiple of the energy needed to reach base.
- **Engine.** A fixed-timestep, headless loop advances UAVs, queries the oracle on cell arrival, and
  emits an event log. It is **deterministic**: a single seeded `numpy` generator threads through the
  oracle and coordinator, so a given seed reproduces a run byte-for-byte. Mid-mission failures are
  injected via a scripted `fail_at` hook.

## 5. Coordination

- **Partitioning** assigns each UAV a sector: a `grid` baseline and a workload-balanced
  `weighted_voronoi` (Lloyd relaxation + greedy boundary rebalance).
- **Coverage** within a sector uses a **boustrophedon** ("lawnmower") sweep, with an explicit
  last-row fix so no strip is skipped. A spiral alternative is implemented for a path-length
  comparison (Results); the lawnmower is the default because it covers *any* sector shape robustly,
  whereas the spiral, though shorter on compact sectors, fails on thin ones.
- **Allocation** is auction-based, in the style of the **Contract Net Protocol** (Smith, 1980),
  classified as ST-SR-IA applied online (Gerkey & Matarić, 2004). Each UAV bids a cost
  `a·travel + b·energy_penalty − c·priority`; the lowest bid wins. On a UAV failure or return-to-home,
  its unfinished cells are re-auctioned. Three baselines share the interface — `single_uav`,
  `static_partition_no_realloc`, `random_walk` — and an `auction_no_priority` ablation isolates the
  reallocation from the priority mechanism. *These methods are established (see related work); the
  contribution is their controlled evaluation, not their invention.*

## 6. Drift model and re-tasking

Survivor drift adapts USCG **SAROPS**: Monte-Carlo particles are advected from the detection point,
each step `dx = u(x,y)·leeway·dt + √(2·K_h·dt)·ξ` (advection + turbulent diffusion), and the cloud is
reduced to 50 %/90 % **containment polygons**. The 90 % polygon maps to grid cells, which raise the
auction priority so UAVs re-task toward the predicted region. Constants (leeway, diffusivity, horizon)
are documented illustrative assumptions in config; the flow field is *assumed*, not estimated from the
imagery (a limitation and a proposed extension).

## 7. Routing

Segmentation hazards fold into a road graph: `road_blocked` removes a cell's edges; `water` /
`building_damaged` raise edge risk. Edge weight is `length·(1 + λ·risk)`; sweeping λ traces the
**Pareto front** of distance vs cumulative risk. The same method runs on a synthetic grid graph and on
a cached real OpenStreetMap network. (Weighted-sum scalarisation recovers only the convex hull of the
front — a documented limitation handled by grading the demo hazard so the front is non-degenerate.)

## 8. Experimental design

All stochastic experiments report **mean ± 95 % confidence interval** over independent seeds.

| Experiment | Command | Design | Primary metric |
|---|---|---|---|
| Coordination under failure (RQ1/2) | `make sweep` | 5 strategies × {1,2,4,6} UAVs × {0,1,2} failures × 30 seeds (1,800 runs) | coverage |
| Drift-aware search (RQ4) | `make rq4` | 300 seeds; true drift vs *independent* prediction | localisation rate & error |
| Perception × coordination (RQ2/5) | `make sensitivity` | FN rate swept 0–0.5 × {auction, static} × 30 seeds under a 2-UAV failure | survivor-detection rate |
| Coverage patterns | `make coverage-compare` | lawnmower vs spiral, path length at equal coverage | path length |
| Hazard-aware routing (RQ6) | `make routes`, `make routes-osm` | λ-sweep Pareto front, synthetic + real OSM | distance vs risk |

Controlled variables (number of UAVs, number of failures, detector FN rate) are varied one dimension
at a time; the failing UAV and failure time are randomised per seed within a fixed window. Baselines
and an ablation accompany the headline comparison so the effect is attributable.

## 9. Reproducibility and tooling

- **Configuration, not magic numbers.** Every constant lives in a Hydra/OmegaConf YAML under
  `configs/` with a provenance comment; `src/` contains no hard-coded parameters.
- **Determinism.** Every stochastic entry point takes an explicit seeded generator; the global RNG is
  never used.
- **Environment.** Python 3.10 with `uv` and a pinned lockfile; a single `make setup` reproduces it.
- **Testing.** 91 automated tests (`make test`, no GPU/datasets) act as the contract, including
  analytic acceptance cases (e.g. zero-diffusion drift = exact v·Δt), a determinism test, and an
  AST import test enforcing ADR-001. `make lint` (ruff) gates every change.
- **Provenance.** Long runs write results and the resolved config to a timestamped `outputs/`
  directory; nothing overwrites a previous run.

## 10. Limitations and ethics

The evaluation is simulation-only; the flow field and several drift constants are assumed rather than
measured; `flood_a` uses synthetic georeferencing and a saturated survivor field; and `person` is
transferred across a domain gap. RescueNet's non-commercial licence is respected (no derived labels
redistributed). These are stated so the results are read within their scope (Results §8).
