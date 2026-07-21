# Reading list

A curated, practical bibliography for this project — ~65 papers grouped by the building block
they support. Each section opens with *why it matters here*. Links are arXiv abstract pages,
DOIs, or (where no stable open link exists) a Google Scholar lookup.

Start with the ★ items in each section — those are the ones you'll cite most.

---

## 1. Small-object & UAV detection — the core perception difficulty

*Survivors are ~0.1% of the frame; this is the hardest part of the perception problem and the
source of your headline result.*

- ★ Akyon et al. (2022) — *Slicing Aided Hyper Inference (SAHI) for Small Object Detection*, ICIP. [arXiv:2202.06934](https://arxiv.org/abs/2202.06934)
- ★ Zhu et al. (2021) — *Detection and Tracking Meet Drones Challenge* (VisDrone), TPAMI. [arXiv:2001.06303](https://arxiv.org/abs/2001.06303)
- Du et al. (2018) — *The Unmanned Aerial Vehicle Benchmark* (UAVDT), ECCV. [arXiv:1804.00518](https://arxiv.org/abs/1804.00518)
- ★ Sambolek & Ivašić-Kos (2021) — *Automatic Person Detection in SAR Operations Using Deep CNN Detectors* (SARD), IEEE Access. [Scholar](https://scholar.google.com/scholar?q=Automatic+Person+Detection+in+Search+and+Rescue+Operations+Using+Deep+CNN+Detectors)
- Božić-Štulić et al. (2019) — *Deep Learning Approach in Aerial Imagery for Supporting Land SAR Missions* (HERIDAL), IJCV. [Scholar](https://scholar.google.com/scholar?q=Deep+Learning+Approach+in+Aerial+Imagery+for+Supporting+Land+Search+and+Rescue+Missions+HERIDAL)
- Kyrkou & Theocharides (2020) — *EmergencyNet: Efficient Aerial Image Classification for Drone-Based Emergency Monitoring* (AIDER), IEEE JSTARS. [Scholar](https://scholar.google.com/scholar?q=EmergencyNet+Efficient+Aerial+Image+Classification+Drone+Emergency+Monitoring)
- Varga et al. (2022) — *SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water*, WACV. [arXiv:2105.01922](https://arxiv.org/abs/2105.01922)

## 2. Object-detection methods & backbones — the ideas YOLO is built on

*Background so you can explain in the viva why a one-stage detector, an FPN, and anchor-free
heads are the right tools, and how the alternatives compare.*

- Ren et al. (2015) — *Faster R-CNN*. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
- Liu et al. (2016) — *SSD: Single Shot MultiBox Detector*. [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)
- ★ Lin et al. (2017) — *Focal Loss for Dense Object Detection* (RetinaNet). [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- ★ Lin et al. (2017) — *Feature Pyramid Networks* (FPN). [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
- Tian et al. (2019) — *FCOS: Fully Convolutional One-Stage Detection*. [arXiv:1904.01355](https://arxiv.org/abs/1904.01355)
- Carion et al. (2020) — *End-to-End Object Detection with Transformers* (DETR). [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- Zhao et al. (2023) — *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR). [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- He et al. (2016) — *Deep Residual Learning* (ResNet). [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- ★ Lin et al. (2014) — *Microsoft COCO* (the size-stratified AP metric you'll report). [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
- Everingham et al. (2010) — *The PASCAL VOC Challenge* (the label format SARD uses). [DOI](https://doi.org/10.1007/s11263-009-0275-4)

## 3. The YOLO family — your Model A/B architecture lineage

*You're using YOLO11; examiners will ask "why this one, and what changed?" These trace the line.*

- Redmon et al. (2016) — *You Only Look Once (YOLOv1)*. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
- Redmon & Farhadi (2017) — *YOLO9000 / YOLOv2*. [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)
- Redmon & Farhadi (2018) — *YOLOv3*. [arXiv:1804.02767](https://arxiv.org/abs/1804.02767)
- Bochkovskiy et al. (2020) — *YOLOv4*. [arXiv:2004.10934](https://arxiv.org/abs/2004.10934)
- Wang et al. (2022) — *YOLOv7*. [arXiv:2207.02696](https://arxiv.org/abs/2207.02696)
- Wang et al. (2024) — *YOLOv9: Programmable Gradient Information*. [arXiv:2402.13616](https://arxiv.org/abs/2402.13616)
- Wang et al. (2024) — *YOLOv10: Real-Time End-to-End Object Detection*. [arXiv:2405.14458](https://arxiv.org/abs/2405.14458)
- ★ Khanam & Hussain (2024) — *YOLOv11: An Overview of the Key Architectural Enhancements*. [arXiv:2410.17725](https://arxiv.org/abs/2410.17725)

## 4. Segmentation — your Model B task (masks, not boxes)

*RescueNet/FloodNet are pixel masks; these cover the segmentation methods behind YOLO11-seg.*

- Long et al. (2015) — *Fully Convolutional Networks* (FCN). [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
- ★ Ronneberger et al. (2015) — *U-Net*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Chen et al. (2017) — *Rethinking Atrous Convolution* (DeepLabv3). [arXiv:1706.05587](https://arxiv.org/abs/1706.05587)
- Chen et al. (2018) — *DeepLabv3+*. [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)
- He et al. (2017) — *Mask R-CNN*. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Cheng et al. (2022) — *Masked-attention Mask Transformer* (Mask2Former). [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)
- Kirillov et al. (2023) — *Segment Anything* (SAM). [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

## 5. Aerial & disaster datasets — your data and its neighbours

*Provenance and benchmarks for the imagery you train on; xBD is the satellite one you skip.*

- Xia et al. (2018) — *DOTA: A Large-scale Dataset for Object Detection in Aerial Images*. [arXiv:1711.10398](https://arxiv.org/abs/1711.10398)
- Waqas Zamir et al. (2019) — *iSAID: Instance Segmentation in Aerial Images*. [arXiv:1905.12886](https://arxiv.org/abs/1905.12886)
- Gupta et al. (2019) — *Creating xBD: A Dataset for Assessing Building Damage* (xView2). [arXiv:1911.09296](https://arxiv.org/abs/1911.09296)
- ★ Rahnemoonfar et al. (2021) — *FloodNet*, IEEE Access. [arXiv:2012.02951](https://arxiv.org/abs/2012.02951)
- ★ Chowdhury et al. (2023) — *RescueNet*, Scientific Data. [arXiv:2202.12361](https://arxiv.org/abs/2202.12361)
- Shamsoshoara et al. (2021) — *Aerial Imagery Pile Burn Detection: The FLAME Dataset*, Computer Networks. [DOI](https://doi.org/10.1016/j.comnet.2021.108001)

## 6. Multi-robot task allocation & auctions — your core contribution

*The re-tasking mechanism. Cite Contract Net + the MRTA taxonomy for "where this came from";
CBBA is the decentralised cousin of your auctioneer.*

- ★ Smith (1980) — *The Contract Net Protocol*, IEEE Trans. Computers. [DOI](https://doi.org/10.1109/TC.1980.1675516)
- ★ Gerkey & Matarić (2004) — *A Formal Analysis and Taxonomy of Task Allocation in Multi-Robot Systems* (ST-SR-TA), IJRR. [DOI](https://doi.org/10.1177/0278364904045564)
- Korsah, Stentz & Dias (2013) — *A Comprehensive Taxonomy for Multi-Robot Task Allocation* (iTax), IJRR. [DOI](https://doi.org/10.1177/0278364913496484)
- Dias et al. (2006) — *Market-Based Multirobot Coordination: A Survey and Analysis*, Proc. IEEE. [DOI](https://doi.org/10.1109/JPROC.2006.876939)
- ★ Choi, Brunet & How (2009) — *Consensus-Based Decentralized Auctions for Robust Task Allocation* (CBBA), IEEE T-RO. [DOI](https://doi.org/10.1109/TRO.2009.2022423)
- Parker (1998) — *ALLIANCE: Fault-Tolerant Multi-Robot Cooperation*, IEEE T-RA. [DOI](https://doi.org/10.1109/70.678498)
- Nanjanath & Gini (2010) — *Repeated Auctions for Robust Task Execution*, Robotics and Autonomous Systems. [DOI](https://doi.org/10.1016/j.robot.2010.05.006)

## 7. Coverage path planning — your survey stage

*How a UAV sweeps a sector without gaps; boustrophedon is what you implement in Phase 5.*

- ★ Choset (2001) — *Coverage for Robotics: A Survey of Recent Results*, Annals of Math & AI. [DOI](https://doi.org/10.1023/A:1016639210559)
- ★ Galceran & Carreras (2013) — *A Survey on Coverage Path Planning for Robotics*, Robotics and Autonomous Systems. [DOI](https://doi.org/10.1016/j.robot.2013.09.004)
- Choset & Pignon (1998) — *Coverage Path Planning: The Boustrophedon Decomposition*. [Scholar](https://scholar.google.com/scholar?q=Coverage+Path+Planning+The+Boustrophedon+Cellular+Decomposition+Choset+Pignon)
- Cabreira, Brisolara & Ferreira (2019) — *Survey on Coverage Path Planning with UAVs*, Drones. [DOI](https://doi.org/10.3390/drones3010004)

## 8. Routing & graph search — your rescue-route stage

*Shortest/safest path on a road graph; OSMnx is the exact tool you'll use in Phase 8.*

- Dijkstra (1959) — *A Note on Two Problems in Connexion with Graphs*, Numerische Mathematik. [DOI](https://doi.org/10.1007/BF01386390)
- ★ Hart, Nilsson & Raphael (1968) — *A Formal Basis for the Heuristic Determination of Minimum Cost Paths* (A*), IEEE Trans. SSC. [DOI](https://doi.org/10.1109/TSSC.1968.300136)
- ★ Boeing (2017) — *OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks*, CEUS. [arXiv:1611.01890](https://arxiv.org/abs/1611.01890) · [DOI](https://doi.org/10.1016/j.compenvurbsys.2017.05.004)
- LaValle (2006) — *Planning Algorithms* (free textbook). [Online](http://lavalle.pl/planning/)

## 9. Search-and-rescue drift / SAROPS — your drift model's lineage

*The drift model adapts maritime SAR planning (leeway + Monte Carlo containment). Cite these
so it reads as "adapted an established method," not "invented a thing."*

- ★ Stone, Royset & Washburn (2016) — *Optimal Search for Moving Targets*, Springer. [DOI](https://doi.org/10.1007/978-3-319-26899-6)
- ★ Kratzke, Stone & Frost (2010) — *Search and Rescue Optimal Planning System* (SAROPS), FUSION/OCEANS. [DOI](https://doi.org/10.1109/ICIF.2010.5711881)
- Breivik & Allen (2008) — *An Operational Search and Rescue Model for the Norwegian Sea* (leeway), J. Marine Systems. [Scholar](https://scholar.google.com/scholar?q=An+operational+search+and+rescue+model+for+the+Norwegian+Sea+Breivik+Allen)
- van Sebille et al. (2018) — *Lagrangian Ocean Analysis: Fundamentals and Practices*, Ocean Modelling. [DOI](https://doi.org/10.1016/j.ocemod.2017.11.008)

## 10. 3D reconstruction — NeRF vs 3DGS vs photogrammetry (Phase 10)

*The comparative study. Read the ISPRS aerial result before starting — photogrammetry often
wins on near-nadir survey imagery, which is a legitimate finding.*

- ★ Mildenhall et al. (2020) — *NeRF: Representing Scenes as Neural Radiance Fields*, ECCV. [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)
- ★ Kerbl et al. (2023) — *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH. [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)
- Barron et al. (2022) — *Mip-NeRF 360*. [arXiv:2111.12077](https://arxiv.org/abs/2111.12077)
- Müller et al. (2022) — *Instant Neural Graphics Primitives* (Instant-NGP). [arXiv:2201.05989](https://arxiv.org/abs/2201.05989)
- Tancik et al. (2023) — *Nerfstudio: A Modular Framework for Neural Radiance Field Development*. [arXiv:2302.04264](https://arxiv.org/abs/2302.04264)
- Turki et al. (2022) — *Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs*. [arXiv:2112.10703](https://arxiv.org/abs/2112.10703)
- Schönberger & Frahm (2016) — *Structure-from-Motion Revisited* (COLMAP). [DOI](https://doi.org/10.1109/CVPR.2016.445)
- ISPRS Annals X-2-2024 — *The Potential of NeRF and 3DGS for 3D Reconstruction from Aerial Imagery* (COLMAP beats Splatfacto on nadir). [Scholar](https://scholar.google.com/scholar?q=Potential+of+NeRF+and+3DGS+for+3D+Reconstruction+from+Aerial+Imagery+ISPRS+2024)

---

## 11. Recent coordination & multi-UAV work (2020–2025) — for the contribution chapter

*The seminal papers above (§6–9) ground the method; these recent ones show you're current and
give you the direct comparison points a strong literature review needs. Read these when writing
the coordination chapter.*

**Multi-UAV task allocation & disaster coordination**
- Reta et al. (2025) — *A distributed task allocation approach for multi-UAV persistent monitoring in dynamic environments*, Scientific Reports. [DOI](https://doi.org/10.1038/s41598-025-89787-3)
- ★ (2023) — *Multi-UAV networks for disaster monitoring: challenges and opportunities*, Drone Systems & Applications. [DOI](https://doi.org/10.1139/dsa-2023-0079)
- (2025) — *Decision-Making-Based Path Planning for Autonomous UAVs: A Survey*. [arXiv:2508.09304](https://arxiv.org/abs/2508.09304)

**Auction / market-based MRTA (recent)**
- ★ Sadeghi & Smith (2023) — *Auction Algorithm Sensitivity for Multi-Robot Task Allocation*. [arXiv:2306.16032](https://arxiv.org/abs/2306.16032)
- (2025) — *Task Allocation in Mobile Robot Fleets: A Review*. [arXiv:2501.08726](https://arxiv.org/abs/2501.08726)
- (2021) — *Greedy Decentralized Auction-based Task Allocation for Multi-Agent Systems*. [arXiv:2107.00144](https://arxiv.org/abs/2107.00144)
- ★ (2020) — *The Application of Market-based Multi-Robot Task Allocation to Ambulance Dispatch*. [arXiv:2003.05550](https://arxiv.org/abs/2003.05550)
- (2025) — *Uncertainty-Aware Multi-Robot Task Allocation With Strongly Coupled Inter-Robot Rewards*. [arXiv:2509.22469](https://arxiv.org/abs/2509.22469)
- (2024) — *Multi-Robot Task Allocation and Path Planning with Maximum Range Constraints*. [arXiv:2409.06531](https://arxiv.org/abs/2409.06531)
- (2023) — *Auction-Based Task Allocation and Motion Planning for Multi-Robot Systems with Human Supervision*, J. Intelligent & Robotic Systems. [DOI](https://doi.org/10.1007/s10846-023-01935-x)

**Learning-based multi-UAV search & rescue** (compare against your auction baseline)
- ★ (2024) — *Deep Reinforcement Learning for Time-Critical Wilderness Search and Rescue Using Drones*. [arXiv:2405.12800](https://arxiv.org/abs/2405.12800)
- (2025) — *DRL-based Autonomous Decision-Making for Cooperative UAVs: A Search-and-Rescue Real-World Application*. [arXiv:2502.20326](https://arxiv.org/abs/2502.20326)
- (2025) — *Recurrent Auto-Encoders for Enhanced Deep RL in Wilderness SAR Planning*. [arXiv:2502.19356](https://arxiv.org/abs/2502.19356)

**Energy-aware coverage** (matches your UAV energy model + RTH)
- ★ (2024) — *Energy-aware Multi-UAV Coverage Mission Planning with Optimal Speed of Flight*. [arXiv:2402.10529](https://arxiv.org/abs/2402.10529)
- (2024) — *Communication and Energy-Aware Multi-UAV Coverage Path Planning for Networked Operations*. [arXiv:2411.02772](https://arxiv.org/abs/2411.02772)

> These give you what the seminal papers can't: modern baselines to position against. Your angle
> — auction reallocation driven by *cached real detections and predicted survivor drift*, scored
> over Monte-Carlo seeds — is not covered by any of them, which is where the novelty sits.

### How to read this efficiently
1. This week: Sections 1–3 (perception) — SAHI, VisDrone, YOLO11, COCO's size-stratified AP.
2. Before Phase 6: Section 6 (auctions) — Contract Net + Gerkey & Matarić are your two must-cite.
3. Before Phase 7/8/10: Sections 9, 8, 10 as you reach them.

Tip: paste any arXiv link into [ar5iv](https://ar5iv.org) (change `arxiv.org` to `ar5iv.org`)
for a clean HTML read, or use a reference manager (Zotero) to import by DOI/arXiv id.
