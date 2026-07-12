import argparse
import json

import joblib
import pandas as pd

from app import config, data, monitoring


def _next_report_path():
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(config.REPORTS_DIR.glob("monitor_*.json"))
    return config.REPORTS_DIR / f"monitor_{len(existing)}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift and performance monitoring on a batch.")
    parser.add_argument("--window", type=str, required=True, help="Path to the incoming batch (parquet or csv).")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the joblib model artifact.")
    parser.add_argument("--train-recall", type=float, required=True, help="Recall on the training-time test set, for trigger comparison.")
    args = parser.parse_args()

    if not config.REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Reference stats not found at {config.REFERENCE_STATS_PATH}. "
            "Run app.train first to generate it."
        )
    with open(config.REFERENCE_STATS_PATH) as f:
        reference_stats = json.load(f)

    model = joblib.load(args.model_path)

    if args.window.endswith(".csv"):
        incoming_df = pd.read_csv(args.window)
    else:
        incoming_df = pd.read_parquet(args.window)

    X_incoming = data.select_features(incoming_df, config.SELECTED_FEATURES)
    y_true = incoming_df[data.TARGET_COLUMN].to_numpy()
    y_pred = model.predict(X_incoming)

    drift = monitoring.drift_report(reference_stats, X_incoming)
    rolling = monitoring.rolling_metrics(
        y_true, y_pred, config.FN_COST, config.FP_COST, window=config.MONITOR_WINDOW_SIZE
    )
    trigger = monitoring.check_retrain_trigger(
        drift, rolling, train_recall=args.train_recall, cost_budget=config.COST_BUDGET
    )

    report = {
        "drift": drift,
        "rolling_metrics": rolling,
        "trigger": trigger,
    }

    report_path = _next_report_path()
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Monitoring report written to {report_path}")
    print(f"Max PSI: {drift['max_psi']:.3f}, drifted features: {drift['drifted_features']}")
    print(f"Retrain triggered: {trigger['triggered']} ({trigger['reasons']})")


if __name__ == "__main__":
    main()
