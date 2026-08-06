"""Non-IID client partitioning by data source — the pure, testable core of the federated setup.

Each data source becomes one federated client (VisDrone = one operator's imagery, SARD = another).
The split is genuinely non-IID and imbalanced, which is the point: it is the setting FedProx is
expected to help over FedAvg. No image copying — each client gets a train.txt listing its own image
paths and a data.yaml that reuses the shared val split, so Ultralytics can train on it directly.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

IMG_EXT = {".jpg", ".jpeg", ".png"}


def group_by_source(filenames, sources) -> dict[str, list[str]]:
    """Group image filenames into clients by their leading ``<source>_`` prefix.

    Pure and dataset-free (this is what the unit test pins). Files whose prefix is not in the list
    are dropped, so an explicit source list controls exactly which clients exist.
    """
    wanted = set(sources)
    groups: dict[str, list[str]] = defaultdict(list)
    for name in filenames:
        prefix = Path(name).name.split("_", 1)[0]
        if prefix in wanted:
            groups[prefix].append(name)
    return {s: sorted(groups[s]) for s in sources if groups.get(s)}


def write_client_datasets(src_data_yaml: Path, out_root: Path, sources) -> dict[str, Path]:
    """Write one data.yaml + train.txt per client from a source-prefixed detect dataset.

    Returns ``{source: client_data_yaml}``. The val split is shared across clients (federated
    evaluation is on the same held-out set); only the train images are partitioned.
    """
    src = OmegaConf.load(src_data_yaml)
    base = Path(src.path)
    train_dir = base / str(src.train)
    names = list(src.names)

    filenames = [p.name for p in train_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    groups = group_by_source(filenames, sources)
    out_root.mkdir(parents=True, exist_ok=True)

    client_yamls: dict[str, Path] = {}
    for source, files in groups.items():
        list_path = out_root / f"{source}_train.txt"
        list_path.write_text("\n".join(str(train_dir / f) for f in files) + "\n")
        yaml_path = out_root / f"{source}.yaml"
        OmegaConf.save(
            OmegaConf.create(
                {
                    "path": str(base),
                    "train": str(list_path),
                    "val": str(src.val),
                    "names": names,
                    "nc": len(names),
                }
            ),
            yaml_path,
        )
        client_yamls[source] = yaml_path
    return client_yamls
