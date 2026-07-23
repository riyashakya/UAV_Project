# Research questions, novelty & related work

MSc dissertation (LSBU) · Student 4437147. This consolidates the research questions, the novel
contributions, and the projects to compare against. Each RQ is mapped to the phase that answers it.

---

## Research questions

**RQ1 (primary — coordination).** Does auction-based *dynamic task reallocation* improve
multi-UAV disaster-survey outcomes — area coverage, time-to-90%-coverage, survivors-found-over-time,
and mission completion under UAV failure — compared with static partitioning and simpler baselines
(single-UAV, random-walk), and under what conditions (UAV count, failure rate)?
→ *Answered (Phase 6 + 9): yes — auction beats static by +12.4 pts (one failure) and +25.5 pts
(two failures) in coverage at 6 UAVs, mean ± 95% CI.*

**RQ2 (sensitivity / robustness).** How sensitive is the reallocation advantage to (a) the number
of UAVs, (b) the failure rate and timing, (c) the bid weights (travel vs energy vs priority), and
(d) perception noise (per-class false-negative rate)?
→ *Partly answered (Phase 9 sweep across UAV count + failure rate); bid-weight and noise sweeps
are a straightforward extension of the existing harness.*

**RQ3 (perception domain gap).** How much accuracy is lost transferring survivor (`person`)
detection from ordinary/SAR imagery to post-disaster imagery, and how much does tiled (SAHI)
inference recover small-object AP versus full-frame inference?
→ *Partly answered (Phase 2b): size-stratified AP (small 0.26 vs large 0.78); per-source gap
(SARD 0.88 vs VisDrone 0.65 AP@50); naive SAHI reduced AP — a documented negative result.*

**RQ4 (drift-driven re-tasking — the most novel).** Does coupling *predicted survivor drift*
(a SAROPS-style leeway + Monte-Carlo containment model) into the reallocation — re-tasking UAVs
toward the predicted search polygon — improve survivor localisation over drift-agnostic
coordination, and how does the advantage scale with current speed and prediction horizon?
→ *Pending (Phase 7).*

**RQ5 (methodology — decoupling).** Does decoupling perception (scored on real labelled data) from
coordination (scored over Monte-Carlo seeds) via a cached detection oracle yield valid, reproducible
design-space exploration, compared with an end-to-end photorealistic simulator?
→ *Supported by the architecture (ADR-001); argued in the methods chapter.*

**RQ6 (hazard-aware routing).** Does folding segmentation hazards (blocked roads, flood water,
damaged buildings) into a hazard-weighted road graph produce rescue routes that trade distance for
safety along a useful Pareto front, versus naive shortest-path routing?
→ *Pending (Phase 8).*

Frame RQ1 as the main question; RQ2–RQ6 as sub-questions/contributions.

---

## Novelty — what this adds that most projects don't

Most existing work does *one* of: disaster-image perception, multi-robot task allocation, or
maritime drift. This project's differentiators:

1. **Drift-driven re-tasking loop** (RQ4) — perception → survivor-drift prediction → coordination,
   closed in one system. Rarely done; the strongest single novelty.
2. **Estimating the water flow field from the drone video itself** — via optical flow of floating
   surface debris — instead of assuming a current. A novel perception↔physics link (extension).
3. **Decoupled, reproducible evaluation** (RQ5) — perception scored on real data, coordination over
   hundreds of CPU Monte-Carlo seeds with confidence intervals. Methodologically rigorous.
4. **Reallocation under realistic failures + detection noise** — the advantage is *measured* with
   CIs, not asserted; sensitivity to noise and comms limits is a clean extension.
5. **Detection-driven priority** — the auction's cell priority comes from the real YOLO detections
   (survivors, hazards), tying perception quality directly to coordination decisions.
6. **Comms-constrained / decentralised auction** (CBBA-style) — dropping the perfect-central-planner
   assumption toward realistic, communication-limited coordination.

---

## Similar projects & systems (for the related-work / comparison chapter)

**Multi-robot / multi-UAV task allocation (the coordination baseline family)**
- Contract Net Protocol (Smith 1980); MRTA taxonomy (Gerkey & Matarić 2004; Korsah et al. 2013).
- CBBA — Consensus-Based Bundle Algorithm (Choi, Brunet & How 2009): decentralised auctions for UAVs.
- Market-based multi-robot coordination survey (Dias et al. 2006); recent 2020–2025 work in
  `reading_list.md` §11 (e.g. distributed multi-UAV persistent monitoring, auction sensitivity).

**Learning-based multi-UAV SAR (the modern alternative to auctions)**
- Deep-RL for time-critical wilderness SAR with drones (2024); cooperative-UAV DRL for SAR (2025).
  These are the natural "learned policy vs classical auction" comparison points.

**Disaster-image perception (the perception baseline family)**
- VisDrone (small-object UAV detection), SARD/HERIDAL (person in SAR), RescueNet/FloodNet
  (post-disaster segmentation) — the datasets and their challenge leaderboards.
- SAHI (Akyon et al. 2022) — the small-object tiled-inference baseline.

**Drift / search-planning (the drift family)**
- USCG **SAROPS** (Kratzke, Stone & Frost 2010) and optimal search theory (Stone et al. 2016);
  leeway models (Breivik & Allen 2008). Built for boats/people in open water — this project adapts
  the method to UAV re-tasking, which the maritime work does not do.

**Simulation platforms & benchmarks (positioning of the sim)**
- AirSim / Gazebo photorealistic UAV sims (what ADR-001 deliberately avoids); RoboCup Rescue
  Simulation (multi-agent disaster-response benchmark); DARPA SubT (multi-robot SAR).

**Positioning statement.** Perception, coordination, and drift each have mature literatures, but
few systems combine UAV perception + adaptive auction coordination + survivor-drift prediction and
evaluate the combination rigorously over Monte-Carlo seeds. That intersection — plus the decoupled
methodology — is this project's niche.
