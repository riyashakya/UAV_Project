"""Build the two unified datasets (ADR-002) from the source datasets — Phase 1.

Dry-run prints a per-class instance-count table for the detect set and the seg set, so the
domain mix is visible before committing to a full conversion. A real run additionally writes
YOLO / YOLO-seg labels (and image symlinks + an Ultralytics ``data.yaml``) under the output
dirs from ``configs/datasets/default.yaml``.

    python -m src.perception.datasets.build --dry-run
    python -m src.perception.datasets.build            # write data/unified/{detect,seg}
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from .loaders import build_loader
from .model import UnifiedImage
from .taxonomy import DEFAULT_TAXONOMY_PATH, UnifiedTaxonomy, load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "datasets" / "default.yaml"


@dataclass
class BuildReport:
    """Instance counts and availability, keyed for both display and assertions."""

    # counts[task][unified_class][source] = number of instances
    counts: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    available: dict[str, bool] = field(default_factory=dict)
    n_images: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def total(self, task: str, unified_class: str) -> int:
        return sum(self.counts[task][unified_class].values())


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def collect(
    tax: UnifiedTaxonomy,
    cfg,
    *,
    sources: list[str] | None = None,
    writer: DatasetWriter | None = None,
) -> BuildReport:
    """Iterate every available source, tally instances, and optionally write labels."""
    report = BuildReport()
    build_cfg = OmegaConf.to_container(cfg.get("build", {}), resolve=True) or {}
    roots = cfg.get("roots", {})

    for name, mapping in tax.sources.items():
        if sources and name not in sources:
            continue
        root = _resolve(roots[name])
        loader = build_loader(name, root, mapping, build_cfg)
        report.available[name] = loader.is_available()
        if not report.available[name]:
            continue

        task = mapping.task
        for image in loader.iter_images():
            report.n_images[name] += 1
            for det in image.detections:
                report.counts[task][det.cls][name] += 1
            for seg in image.segments:
                report.counts[task][seg.cls][name] += 1
            if writer is not None:
                writer.write(image, task)

    return report


def render_table(report: BuildReport, tax: UnifiedTaxonomy) -> str:
    """Render the dry-run instance-count table for both output sets."""
    lines: list[str] = []
    present = [s for s, ok in report.available.items() if ok]
    absent = [s for s, ok in report.available.items() if not ok]

    for set_name, task in (("DETECT (Model A)", "detect"), ("SEGMENT (Model B)", "segment")):
        classes = tax.classes(task)
        sources = [n for n, m in tax.sources.items() if m.task == task]
        header = f"{'class':<20}" + "".join(f"{s:>12}" for s in sources) + f"{'TOTAL':>12}"
        lines.append(f"\n{set_name}")
        lines.append("-" * len(header))
        lines.append(header)
        lines.append("-" * len(header))
        for cls in classes:
            row = f"{cls:<20}"
            for s in sources:
                row += f"{report.counts[task][cls].get(s, 0):>12,}"
            row += f"{report.total(task, cls):>12,}"
            lines.append(row)
        lines.append("-" * len(header))

    lines.append("")
    lines.append(f"images seen : {dict(report.n_images) or '(none)'}")
    lines.append(f"present     : {present or '(none — no datasets on disk yet)'}")
    if absent:
        lines.append(f"absent      : {absent}  (paths from configs/datasets/default.yaml)")
    return "\n".join(lines)


class DatasetWriter:
    """Write unified images to YOLO / YOLO-seg on disk (labels + image symlinks + data.yaml)."""

    def __init__(self, tax: UnifiedTaxonomy, detect_dir: Path, seg_dir: Path):
        self.tax = tax
        self.dirs = {"detect": Path(detect_dir), "segment": Path(seg_dir)}
        for task, base in self.dirs.items():
            (base / "images").mkdir(parents=True, exist_ok=True)
            (base / "labels").mkdir(parents=True, exist_ok=True)

    def write(self, image: UnifiedImage, task: str) -> None:
        base = self.dirs[task]
        stem = f"{image.source}_{Path(image.image_path).stem}"
        label_path = base / "labels" / f"{stem}.txt"

        rows: list[str] = []
        if task == "detect":
            for det in image.detections:
                cid = self.tax.class_id("detect", det.cls)
                x, y, w, h = det.bbox.as_yolo()
                rows.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        else:
            for seg in image.segments:
                cid = self.tax.class_id("segment", seg.cls)
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in seg.polygon)
                rows.append(f"{cid} {coords}")
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""))

        src_img = Path(image.image_path)
        if src_img.exists() and src_img.suffix.lower() in (".jpg", ".jpeg", ".png"):
            link = base / "images" / f"{stem}{src_img.suffix}"
            if not link.exists():
                os.symlink(src_img.resolve(), link)

    def write_data_yaml(self) -> None:
        for task, base in self.dirs.items():
            names = list(self.tax.classes(task))
            cfg = OmegaConf.create(
                {
                    "path": str(base.resolve()),
                    "train": "images",
                    "val": "images",
                    "nc": len(names),
                    "names": names,
                }
            )
            OmegaConf.save(cfg, base / "data.yaml")


def run(argv: list[str] | None = None) -> BuildReport:
    """Programmatic entry point; returns the :class:`BuildReport` (used by tests)."""
    parser = argparse.ArgumentParser(description="Build the unified detect/seg datasets.")
    parser.add_argument("--dry-run", action="store_true", help="count instances, write nothing")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="datasets config YAML")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH), help="taxonomy YAML")
    parser.add_argument("--sources", nargs="*", default=None, help="restrict to these sources")
    args = parser.parse_args(argv)

    tax = load_taxonomy(args.taxonomy)
    cfg = OmegaConf.load(args.config)

    writer = None
    if not args.dry_run:
        out = cfg.get("output", {})
        writer = DatasetWriter(tax, _resolve(out["detect_dir"]), _resolve(out["seg_dir"]))

    report = collect(tax, cfg, sources=args.sources, writer=writer)
    if writer is not None:
        writer.write_data_yaml()

    print(render_table(report, tax))
    if args.dry_run:
        print("\n[dry-run] no files written.")
    else:
        print("\nWrote unified datasets to data/unified/{detect,seg}.")
    return report


def main() -> None:
    run()


if __name__ == "__main__":
    main()
