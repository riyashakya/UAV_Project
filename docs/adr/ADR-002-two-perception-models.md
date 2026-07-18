# ADR-002: Two perception models, not one

- **Status:** Accepted
- **Date:** 2026-07-17
- **Deciders:** Student 4437147 (with supervisor Anastasios Dagiuklas)

## Context

The system must detect five target concepts: `person`, `vehicle`, `building_damaged`,
`road_blocked`, and `water`. No single public dataset covers all five, and — critically —
the source label *types* differ:

- `person` / `vehicle` come from datasets annotated with **bounding boxes** (VisDrone, SARD).
- `building_damaged` / `road_blocked` / `water` come from datasets annotated with **pixel
  masks** (RescueNet, FloodNet).

A flood region has no meaningful bounding box: a box around standing water is neither
localisation nor extent. Forcing everything into one detector would either throw away the
mask information or fabricate boxes that mean nothing.

## Decision

Train and run **two** YOLO11 models with distinct tasks and taxonomies:

| Model | Task | Classes | Source datasets |
|---|---|---|---|
| **A** (`yolo11*.pt`) | detect | `person`, `vehicle` | VisDrone + SARD (+ HERIDAL/UAVDT if needed) |
| **B** (`yolo11*-seg.pt`) | segment | `building_damaged`, `road_blocked`, `water` | RescueNet + FloodNet |

- Model A is **always** run with SAHI tiled inference (640×640 slices, `overlap_ratio=0.2`)
  because survivors are ~0.1 % of frame area (see the "small objects" note in CLAUDE.md).
- Model B consumes RescueNet/FloodNet masks converted to **YOLO-seg polygons**. Masks are
  never reduced to bounding boxes.

## Consequences

**Positive**
- Each model is trained on label types it can actually learn from.
- Extent-based hazards (`water`, `building_damaged`, `road_blocked`) retain their shape,
  which the routing layer (Phase 8) needs to raise edge risk near hazards.

**Negative / accepted trade-offs**
- Two training/eval pipelines and two config trees instead of one.
- Some source classes appear in both worlds (e.g. RescueNet `vehicle` is a *mask*). Phase 1
  emits `vehicle` from box datasets only; mask→bbox for RescueNet `vehicle` is deferred and
  documented in `docs/datasets.md`.

## Related

- Unified taxonomy and the full per-source class mapping live in
  `configs/datasets/taxonomy.yaml`; totality (no silent class drops) is enforced by
  `tests/test_taxonomy.py`.
- See ADR-001 for why neither model is ever called from `src/sim/`.
