from __future__ import annotations

import pandas as pd

from ..io_paths import Paths


def main() -> None:
    p = Paths.from_project_root()

    if not p.feature_store_csv.exists():
        raise FileNotFoundError(f"Missing feature store: {p.feature_store_csv}. Run Stage 1 first.")

    fs = pd.read_csv(p.feature_store_csv)

    print("")
    print("=== FEATURE STORE SANITY SUMMARY (FROM DISK) ===")
    print(f"Rows: {len(fs):,}")
    print(f"Date range: {fs['date'].min()} -> {fs['date'].max()}")

    pct_with_headlines = (fs["headlines_count"] > 0).mean() * 100.0
    print(f"% days with headlines_count > 0: {pct_with_headlines:.2f}%")

    nonzero = fs.loc[fs["headlines_count"] > 0, "sentiment_mean"]
    if len(nonzero) > 0:
        print(
            "sentiment_mean (days with headlines) "
            f"min/median/max: {nonzero.min():.4f} / {nonzero.median():.4f} / {nonzero.max():.4f}"
        )
    else:
        print("sentiment_mean: no days with headlines > 0 in this window.")

    corr_sm = fs["sentiment_mean"].corr(fs["target_next_return"])
    corr_hc = fs["headlines_count"].corr(fs["target_next_return"])
    print(f"corr(sentiment_mean, next_return): {corr_sm:.4f}")
    print(f"corr(headlines_count, next_return): {corr_hc:.4f}")
    print("==============================================")
    print("")


if __name__ == "__main__":
    main()