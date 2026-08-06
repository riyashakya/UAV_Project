"""Federated detector training with Flower — SCAFFOLD (set up, not yet run).

Runs simulated federated training over the non-IID source clients (VisDrone, SARD) with FedAvg or
FedProx aggregation, to compare against the centrally-trained detector. This is a starting scaffold:
the partitioning (`partition.py`) and the FedAvg averaging (`fedavg.py`) are tested, but the Flower
loop below has **not** been executed here — it needs the optional dependency and a GPU:

    uv sync --extra federated
    make federated-train        # strategy/rounds/model in configs/perception/federated.yaml

Honest limitations, written down so they are not overstated:
  * federated learning for UAV detection with FedAvg/FedProx is established — this is a comparison
    in our setting, not a new method (see docs/federated_plan.md);
  * the FedProx **proximal term** is not yet added to Ultralytics' closed training loop, so the
    FedProx path currently differs from FedAvg only in server aggregation, not in the client loss —
    wiring the proximal regulariser into the local optimiser is the main remaining work;
  * a small model (yolo11n) and few rounds are the sensible starting point for compute.

Perception-only (ADR-001). All heavy imports are lazy so unit-test collection stays dependency-free.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.perception.federated.partition import write_client_datasets

REPO_ROOT = Path(__file__).resolve().parents[3]


def _pick_device() -> str | int:
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_weights(model):
    """YOLO model -> list of ndarrays (Flower's NumPyClient parameter format)."""
    return [v.cpu().numpy() for v in model.model.state_dict().values()]


def _set_weights(model, weights) -> None:
    """Load a list of ndarrays back into the YOLO model's state dict (key order preserved)."""
    import torch

    sd = model.model.state_dict()
    new_sd = {k: torch.tensor(w) for k, w in zip(sd.keys(), weights)}
    model.model.load_state_dict(new_sd, strict=True)


def make_client(client_yaml: Path, cfg):
    """Build a Flower NumPyClient that locally trains a YOLO model on one source's data."""
    import flwr as fl
    from ultralytics import YOLO

    device = _pick_device()

    class YoloClient(fl.client.NumPyClient):
        def __init__(self):
            self.model = YOLO(str(cfg.model))
            self.n = sum(1 for _ in Path(client_yaml).with_suffix(".txt").open())

        def get_parameters(self, config):
            return _get_weights(self.model)

        def fit(self, parameters, config):
            _set_weights(self.model, parameters)
            self.model.train(
                data=str(client_yaml),
                epochs=int(cfg.local_epochs),
                imgsz=int(cfg.imgsz),
                batch=int(cfg.batch),
                device=device,
                seed=int(cfg.seed),
                verbose=False,
                plots=False,
            )
            return _get_weights(self.model), self.n, {}

        def evaluate(self, parameters, config):
            _set_weights(self.model, parameters)
            m = self.model.val(
                data=str(client_yaml),
                imgsz=int(cfg.imgsz),
                device=device,
                verbose=False,
                plots=False,
            )
            return float(1.0 - m.box.map50), self.n, {"map50": float(m.box.map50)}

    return YoloClient().to_client()


def main() -> None:
    import flwr as fl

    cfg = OmegaConf.load(REPO_ROOT / "configs/perception/federated.yaml")
    out_root = REPO_ROOT / str(cfg.out)
    clients = write_client_datasets(REPO_ROOT / cfg.data, out_root / "clients", list(cfg.sources))
    ids = list(clients)
    print(
        f"[federated] clients={ {k: str(v.name) for k, v in clients.items()} } "
        f"strategy={cfg.strategy} rounds={cfg.rounds}"
    )

    def client_fn(context):  # Flower passes a context with the partition id
        cid = int(context.node_config.get("partition-id", 0)) % len(ids)
        return make_client(clients[ids[cid]], cfg)

    if str(cfg.strategy).lower() == "fedprox":
        strategy = fl.server.strategy.FedProx(proximal_mu=float(cfg.proximal_mu))
    else:
        strategy = fl.server.strategy.FedAvg()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(ids),
        config=fl.server.ServerConfig(num_rounds=int(cfg.rounds)),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
