"""End to end through the one-command CLI on a synthetic Shopify brand."""
import pandas as pd
import yaml

from src.run import main
from tests.conftest import write_shopify_export


def test_clean_fit_predict_backtest(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_shopify_export(raw / "orders_export.csv", start="2022-01-03", days=800)
    cfg = yaml.safe_load(open("configs/shopify_example.yaml"))
    cfg["paths"] = {k: str(tmp_path / k) for k in ("raw_dir", "processed_dir", "reports_dir", "outputs_dir", "artifacts_dir")}
    cfg["paths"]["raw_dir"] = str(raw)
    cfg["backtest"]["folds"] = {"F1": "2023-10-02"}  # needs 1 year + 2 x 13 weeks of history first
    p = tmp_path / "brand.yaml"
    p.write_text(yaml.safe_dump(cfg))

    main(["--config", str(p), "--stage", "clean"])
    assert (tmp_path / "processed_dir" / "standard.parquet").exists()
    assert (tmp_path / "reports_dir" / "data_summary.md").exists()

    main(["--config", str(p), "--stage", "fit"])
    arts = list((tmp_path / "artifacts_dir").glob("model_*"))
    assert len(arts) == 1

    main(["--config", str(p), "--stage", "predict", "--horizon", "4"])
    out = list((tmp_path / "outputs_dir").glob("forecast_*.csv"))
    fc = pd.read_csv(out[0])
    assert list(fc.columns) == ["unique_id", "ds", "p50", "p10", "p90"] and len(fc) == 6 * 4

    main(["--config", str(p), "--stage", "backtest"])
    assert (tmp_path / "reports_dir" / "engine_backtest.md").exists()
