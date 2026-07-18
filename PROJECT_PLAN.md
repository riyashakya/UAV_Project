# Execution plan — Multi-UAV AI framework for disaster management

Student 4437147 · Supervisor: Anastasios Dagiuklas · Simulation-based

---

## 1. Read this first: a scope warning

Your approved proposal has six objectives spanning multi-agent coordination, object
detection, multi-object tracking, NeRF, 3D Gaussian Splatting, GIS routing, and a
risk/limitations analysis. Your Claude Code prompt then **adds a seventh** — flood drift
prediction — which does not appear in the proposal at all.

Any one of objectives 1–5 is a defensible MSc dissertation on its own. All six plus
drift, done properly, is roughly a PhD's first year.

This is fixable and it is much better to fix it now than in week 10. Two things to do:

1. **Email Dagiuklas about the drift model before you build it.** It is the most novel
   idea in your prompt — coordination + YOLO + NeRF + GIS is all well-trodden ground, but
   *"UAV re-tasking driven by predicted survivor drift"* is a genuine contribution and
   would sharpen the whole project. But it is not in your approved proposal. Get it added
   explicitly, or drop it. Do not smuggle it in.
2. **Agree a descope ladder with him now** (§6 below), so that cutting scope in week 9 is
   executing a plan rather than admitting failure.

The plan below assumes you keep everything, and tells you what to cut first.

---

## 2. The one architectural decision that saves you weeks

**Do not build a photorealistic simulator. Decouple perception from coordination.**

The obvious reading of your prompt is: build a simulated world, fly drones in it, run
YOLO on the rendered camera feed. This is the trap. It means AirSim or Gazebo, an Unreal
build, days of environment setup — and at the end your YOLO numbers are measured on
*synthetic imagery*, which tells an examiner nothing about real detection performance.

Instead:

```
   OFFLINE (GPU, once)                    SIM (CPU, 500× in minutes)
   ┌────────────────────┐                 ┌──────────────────────┐
   │ Real labelled UAV  │                 │ Grid world + flow    │
   │ imagery            │                 │ field                │
   │  ↓                 │                 │  ↓                   │
   │ YOLO11 A + B       │  detections     │ 4 UAVs, energy model │
   │  ↓                 │  ──────────→    │  ↓                   │
   │ detections.parquet │  (oracle.py)    │ partition → survey → │
   │  ↓                 │                 │ auction realloc      │
   │ mAP vs ground truth│                 │  ↓                   │
   └────────────────────┘                 │ drift → routing      │
                                          └──────────────────────┘
   Contribution 1: perception             Contribution 2: coordination
   scored on REAL data                    scored over 500 MC seeds
```

You get two clean contributions instead of one muddy one. Coordination experiments run on
your laptop with no GPU. And when the HPC queue is full in week 8, you are not blocked.

---

## 3. Datasets

### 3.1 The finding that matters most

**No disaster dataset has survivors labelled.** RescueNet, FloodNet and xBD all annotate
buildings, roads, water, vehicles and trees — none has a `person` class. Meanwhile every
person-from-UAV dataset (SARD, HERIDAL, VisDrone) is shot in ordinary conditions, not
disaster zones.

So the single most important object in your system — a survivor — cannot be trained on
in-domain data. There is no dataset that fixes this. What you do instead:

- Train `person` on SARD + VisDrone, accept the domain gap, and **measure and report it**
  rather than hiding it. "Survivor detection transfers from SAR imagery to post-hurricane
  imagery with X% AP drop" is a real finding.
- SARD is your best proxy: its actors deliberately pose as *exhausted or injured* people,
  including lying prone. That is closer to a survivor than VisDrone's walking pedestrians.
- Say this explicitly in your limitations chapter. An examiner who spots this gap when you
  haven't flagged it will treat it as a hole; flagged, it's evidence you understand your
  domain.

### 3.2 Core datasets — get these

| Dataset | What it gives you | Size / format | Notes |
|---|---|---|---|
| **VisDrone2019-DET** | `person`, `vehicle` from UAV altitude | 6,471 train / 548 val / 1,610 test-dev, 10 classes, YOLO format | Auto-downloads via Ultralytics `data="VisDrone.yaml"` (~2 GB). **68% of objects are <32×32 px.** Start here. |
| **SARD** | `person` in injured/exhausted poses | 1,981 images, 1920×1080, ~6,525 person instances, VOC XML | IEEE DataPort. Frames from a 35-min UAV video; poses labelled standing/walking/running/sitting/lying. `Corr` extension adds fog/snow/blur. Your best survivor proxy. |
| **RescueNet** | `building_damaged`, `road_blocked`, `water`, `vehicle` | 4,494 images, 3000×4000, 11 mask classes | **Your most valuable dataset.** DJI Mavic Pro, post-Hurricane Michael. Has `Road-Blocked` and `Building-Total-Destruction` — exactly your classes. Licence **CC BY-NC-ND**: use it, don't redistribute derived labels. |
| **FloodNet** | flood-specific `water`, `building-flooded`, `road-flooded` | ~2,343 images, 3000×4000, 10 mask classes | Post-Hurricane Harvey (Texas). Same lab as RescueNet (BinaLab). Use for the flood scenario. |

### 3.3 Optional — add only if a specific gap bites

| Dataset | When to bother |
|---|---|
| **HERIDAL** | More wilderness `person` data. ~500 labelled 4000×3000 images, 3,229 person annotations, Croatian/Bosnian terrain. Person ≈ 0.1% of frame. |
| **SeaDronesSee** | If you build the flood-drift scenario. People *in water*, 54k+ frames, ~400k instances — and crucially it ships **altitude metadata (5–260 m) and viewing angle (0–90°)**, letting you stratify AP by altitude. That's a strong evaluation angle few students do. |
| **WiSARD** | 33,786 RGB + 22,156 thermal labelled images, 15,453 synced pairs. Only if you go multi-modal. |
| **UAVDT** | 23k train / 15k test, vehicles only (car/truck/bus). Extra vehicle data if `vehicle` AP is weak. |
| **FLAME / FLAME2** | Only if you add a wildfire scenario. |
| **xBD** | **Probably skip.** It's *satellite* imagery (~0.8 m/px, nadir). The domain gap to UAV imagery is severe. Your proposal lists it; I'd use it for pretraining at most, and say why in the write-up. |

### 3.4 Unified taxonomy

```
Model A — detect        Model B — segment
  0 person                0 building_damaged
  1 vehicle               1 road_blocked
                          2 water
```

Mapping:

| Target | Source classes |
|---|---|
| `person` | VisDrone `pedestrian`+`people`; SARD all poses; HERIDAL `person` |
| `vehicle` | VisDrone `car`+`van`+`truck`+`bus`; RescueNet `Vehicle`; UAVDT all |
| `building_damaged` | RescueNet `Building-Major-Damage`+`Building-Total-Destruction`; FloodNet `building-flooded` |
| `road_blocked` | RescueNet `Road-Blocked`; FloodNet `road-flooded` |
| `water` | RescueNet `Water`; FloodNet `water` |

Two notes:

- **"Broken bridges" has no dataset.** Nothing public labels bridge damage from UAV
  imagery at usable scale. Fold it into `road_blocked` and say so, or hand-label ~200
  images from Roboflow Universe if you have spare time (you won't).
- **Don't box the water.** RescueNet's `water` is a mask; a bounding box around a flood
  region is meaningless. Hence two models (ADR-002), not one.

### 3.5 SAHI is not optional

Your survivors are ~0.1% of frame area. Standard YOLO inference resizes a 3000×4000 image
to 640×640 and the survivor stops existing.

Slicing Aided Hyper Inference (Akyon et al., ICIP 2022) tiles the image into overlapping
640×640 slices, infers per-slice, and merges. On VisDrone and xView the original paper
reports **+6.8 / +5.1 / +5.3 AP** for FCOS / VFNet / TOOD, rising to **+12.7 / +13.4 /
+14.5 AP** with slicing-aided fine-tuning. Ultralytics documents the YOLO11 integration
directly.

The full-frame-vs-SAHI ablation on small objects is one of the cleanest results you will
produce. Budget a day, get a chapter figure.

---

## 4. Objective 4 (NeRF / 3DGS): a problem you should know about now

Two things make this the riskiest objective in the proposal.

**(a) Your datasets can't be used for it.** FloodNet and RescueNet are *individually
annotated frames*, not contiguous flight sequences with view overlap. NeRF and 3DGS need
many overlapping views of the same scene plus COLMAP-derived camera poses. You cannot run
Splatfacto on RescueNet. You will need a different data source: a public UAV
photogrammetry sequence (UseGeo, UrbanScene3D), or your own footage.

**(b) Nadir survey imagery is the hard case for these methods.** Disaster survey flights
are near-nadir: large overlap, very short baselines, near-identical viewing directions,
roughly constant standoff distance. This is the opposite of the object-centric orbit that
NeRF and 3DGS are tuned for. A 2024 ISPRS study comparing COLMAP, Nerfacto and Splatfacto
on the UseGeo aerial dataset found **traditional COLMAP still outperformed both** on
less-textured areas, high vegetation, shadowed regions, and areas seen from few views.

This is good news if you reframe it. Don't set out to show NeRF/3DGS produces nice
disaster reconstructions — you may well find it doesn't, and then you have nothing.
Instead:

> **"Under what capture geometry does 3DGS beat photogrammetry for UAV disaster survey?"**

Run COLMAP/WebODM, Nerfacto and Splatfacto on the same scene; report PSNR/SSIM/LPIPS,
wall-clock, VRAM; characterise where each wins. A well-evidenced negative result marks
perfectly well. A missing chapter does not.

---

## 5. Timeline

Assumes ~14 weeks. **Map this to your actual submission date before you start** — if you
have 10 weeks, cut from §6 immediately rather than compressing everything.

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Scaffold, datasets downloaded, taxonomy fixed | Phase 0–1. `docs/datasets.md` |
| 2 | YOLO11 baseline training | Phase 2. First mAP numbers |
| 3 | SAHI ablation, size-stratified AP, domain-gap study | Perception chapter figures |
| 4 | Detection cache, oracle | Phase 3. GPU work done — everything after is CPU |
| 5 | Sim core: world, UAV, energy model | Phase 4 |
| 6 | Partitioning + coverage paths | Phase 5 |
| 7–8 | **Auction reallocation + baselines** | Phase 6. Your core contribution — give it two weeks |
| 9 | Evaluation harness + first MC sweep | Phase 9. **Go/no-go on remaining scope** |
| 10 | Drift model | Phase 7 |
| 11 | Routing + Pareto fronts | Phase 8 |
| 12 | 3D reconstruction study | Phase 10 |
| 13–14 | Writing, final sweeps, limitations chapter | Dissertation |

Two structural points:

- **Evaluation at week 9, not week 13.** Build the harness the moment coordination works.
  Everything after week 9 is then measurable the day it lands. Students who leave
  evaluation to the end discover in week 13 that their headline result doesn't hold.
- **All GPU work finishes by week 4.** After that, HPC queue times can't block you.

---

## 6. Descope ladder — agree this with your supervisor now

Cut in this order. Each cut is defensible in the write-up if you explain the reasoning.

1. **Tracking (ByteTrack/DeepSORT/BoT-SORT).** Detection is what feeds your system;
   tracking adds a metric suite (MOTA/IDF1/HOTA) for little architectural gain. Cut first.
2. **NeRF.** Keep 3DGS only — it's faster to train and the more current method. Report the
   comparison as photogrammetry vs 3DGS.
3. **3D reconstruction entirely.** Replace with a WebODM photogrammetry baseline and a
   literature-based discussion of why NeRF/3DGS underperform on nadir capture. Cite the
   ISPRS UseGeo result.
4. **Multi-scenario evaluation.** One scenario (flood, FloodNet) done thoroughly beats
   three done thinly.
5. **Drift model** — *only* if it never made it into the approved proposal.

**Never cut:** the coordination baselines, the Monte Carlo seeds, or the ablations. Those
*are* the contribution. A system with no baseline to compare against is a demo, not
research.

---

## 7. Risk register

| Risk | L | I | Mitigation |
|---|---|---|---|
| Scope (7 objectives, 1 student) | High | High | §6 ladder agreed with supervisor in week 1 |
| Survivor domain gap (no in-domain person labels) | **Certain** | Med | Measure it, report it, own it in limitations |
| Small-object AP collapses at altitude | High | High | SAHI + size-stratified reporting from week 3 |
| NeRF/3DGS lose to photogrammetry on nadir data | Med-High | Low *if reframed* | Reframe as comparative study (§4) |
| RescueNet/FloodNet unusable for 3D (no view overlap) | **Certain** | Med | Separate data source for objective 4; plan it now |
| HPC queue blocks final runs | Med | High | All GPU work by week 4; Colab Pro for dev |
| Reallocation shows no gain over static partitioning | Med | Med | Still a result. Ablations explain *why*. |
| RescueNet CC BY-NC-ND licence | Low | Med | Non-commercial academic use fine; don't redistribute derived labels |
| Sim-to-real gap questioned in viva | High | Low | Pre-empt it: simulation is for design-space exploration, not deployment claims. Say it first. |

---

## 8. Checklist

### Setup
- [ ] Email Dagiuklas re: drift model scope + descope ladder
- [ ] Confirm actual submission date; re-map §5
- [ ] LSBU HPC access requested and **tested with a hello-world GPU job** (not just requested)
- [ ] Repo + `CLAUDE.md` + ADRs committed
- [ ] `make setup && make test && make lint` green

### Data
- [ ] VisDrone auto-download verified
- [ ] SARD obtained (IEEE DataPort)
- [ ] RescueNet obtained (BinaLab GitHub → Dropbox/figshare)
- [ ] FloodNet obtained
- [ ] Unified taxonomy implemented; mapping test passes (no silent drops)
- [ ] `docs/datasets.md` written with licence + citation per source

### Perception
- [ ] Model A trained; mAP@50, mAP@50-95, per-class AP recorded
- [ ] Model B (seg) trained
- [ ] **SAHI vs full-frame ablation on small objects** ✱ chapter figure
- [ ] AP stratified by COCO object size
- [ ] Domain-gap study: VisDrone-only vs SARD-only vs combined ✱ limitations material
- [ ] Detection cache built; oracle deterministic under seed
- [ ] AST test: `src/sim/` cannot reach `ultralytics`

### Coordination
- [ ] Energy model + RTH threshold, tested
- [ ] Grid and weighted-Voronoi partitioning
- [ ] Boustrophedon coverage >99% cell visitation
- [ ] Auction reallocation working
- [ ] Battery-abort handoff test passes; static baseline **fails** the same test
- [ ] Three baselines implemented (single / static / random)
- [ ] Ablations: realloc off, priority off

### Drift & routing *(scope-dependent)*
- [ ] Advection: analytic test (uniform flow → exactly v·Δt)
- [ ] MC containment: 90% polygon contains ≥90% of particles over 1,000 seeds
- [ ] SAROPS lineage cited in docstring and write-up
- [ ] OSMnx graph cached to disk
- [ ] `road_blocked` edges provably never traversed
- [ ] Pareto front (distance vs risk) with naive baseline overlaid ✱ chapter figure

### Evaluation
- [ ] Harness built **by week 9**
- [ ] Sweep: 4 baselines × N∈{1,2,4,6} × 30 seeds × scenarios
- [ ] All results reported mean ± 95% CI
- [ ] Test split touched **once**, at the end
- [ ] Every headline claim traceable to a run directory with its resolved config

### Write-up
- [ ] All six proposal objectives addressed — including honestly reporting the ones descoped
- [ ] Limitations chapter covers: survivor domain gap, sim-to-real, nadir/3DGS, dataset licences
- [ ] Reproducibility: seeds, configs, and commit hash for every figure
- [ ] Ethics: UAV surveillance, imagery of disaster victims, dual-use

✱ = highest-value figures. If you produce only four plots, produce these four.

---

## 9. Key references to have ready

- Akyon et al. (2022), *Slicing Aided Hyper Inference and Fine-tuning for Small Object
  Detection*, ICIP — arXiv:2202.06934
- Rahnemoonfar et al. (2021), *FloodNet*, IEEE Access
- Chowdhury et al. (2023), *RescueNet*, Scientific Data 10 — nature.com/articles/s41597-023-02799-4
- Sambolek & Ivašić-Kos (2021), *Automatic Person Detection in SAR Operations Using Deep
  CNN Detectors*, IEEE Access (SARD)
- Božić-Štulić et al. (2019), *Deep Learning Approach in Aerial Imagery for Supporting
  Land Search and Rescue Missions*, IJCV (HERIDAL)
- Zhu et al. (2021), *Detection and Tracking Meet Drones Challenge*, TPAMI (VisDrone)
- Gerkey & Matarić (2004), *A Formal Analysis and Taxonomy of Task Allocation in
  Multi-Robot Systems*, IJRR — your ST-SR-TA framing
- Smith (1980), *The Contract Net Protocol*, IEEE Trans. Computers — your auction mechanism
- Kerbl et al. (2023), *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH
- ISPRS Annals X-2-2024, *The Potential of NeRF and 3DGS for 3D Reconstruction from Aerial
  Imagery* — the COLMAP-beats-Splatfacto-on-nadir result
- Choset (2001), *Coverage for Robotics: A Survey of Recent Results* — boustrophedon decomposition
