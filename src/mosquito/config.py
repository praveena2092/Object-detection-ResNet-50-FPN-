from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import yaml

CLASSES = ["aegypti", "albopictus", "anopheles", "culex", "culiseta", "japonicus/koreicus"]
NUM_CLASSES = len(CLASSES)  # torchvision adds background as label 0, so model uses NUM_CLASSES + 1


@dataclass
class Config:
    data_root: str = "data"
    sample_submission: str = "data/sample_submission.csv"
    out_dir: str = "outputs"
    val_fraction: float = 0.15
    batch_size: int = 4
    epochs: int = 10
    lr: float = 0.005
    weight_decay: float = 0.0005
    num_workers: int = 2
    seed: int = 42
    enhance_low_light: bool = True
    hflip: bool = True
    score_sweep: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7])
    score_thresh: float = 0.5

    @property
    def train_img(self) -> str:
        return str(Path(self.data_root) / "train" / "images")

    @property
    def train_lbl(self) -> str:
        return str(Path(self.data_root) / "train" / "labels")

    @property
    def test_img(self) -> str:
        return str(Path(self.data_root) / "test" / "images")


def load_config(path: Optional[str] = None, **overrides) -> Config:
    """Load YAML (if given), then apply non-None keyword overrides."""
    values = {}
    if path:
        with open(path) as f:
            values = yaml.safe_load(f) or {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    valid = {f.name for f in fields(Config)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**values)
