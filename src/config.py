"""Load cost assumptions from config/costs.yaml."""
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "costs.yaml"


@dataclass(frozen=True)
class CostConfig:
    c_r: float    # cost per manual review
    rho: float    # review catch rate for true fraud
    phi: float    # overhead per approved fraud (on top of amount)
    m: float      # margin fraction lost on a false decline
    kappa: float  # fixed cost per false decline
    delta: float  # friction cost of reviewing a legit txn
    K: float      # review capacity as a fraction of transactions
    seed: int = 42

    def with_(self, **kw) -> "CostConfig":
        return replace(self, **kw)


def load_config(path: Path = CONFIG_PATH) -> CostConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return CostConfig(**raw)
