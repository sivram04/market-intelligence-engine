from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    # src/marketintel/io_paths.py -> src/marketintel -> src -> project root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    outputs_dir: Path
    reports_dir: Path

    market_features_csv: Path
    gdelt_headlines_csv: Path
    daily_sentiment_csv: Path
    feature_store_csv: Path

    @staticmethod
    def from_project_root() -> "Paths":
        root = project_root()
        data_dir = root / "data"
        outputs_dir = root / "outputs"
        reports_dir = root / "reports"

        return Paths(
            root=root,
            data_dir=data_dir,
            outputs_dir=outputs_dir,
            reports_dir=reports_dir,
            market_features_csv=data_dir / "market_features.csv",
            gdelt_headlines_csv=data_dir / "gdelt_headlines.csv",
            daily_sentiment_csv=data_dir / "daily_sentiment_gdelt.csv",
            feature_store_csv=data_dir / "feature_store.csv",
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)