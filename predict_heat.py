import os
import argparse
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sequenced_data import windowed_test


sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})


def get_latest_model_path(base_dir: str = "weights") -> str:
    """Return the path to the latest saved model."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"{base_dir} does not exist")
    candidates = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No models found in {base_dir}")
    latest_dir = sorted(candidates)[-1]
    return os.path.join(base_dir, latest_dir, "whole_model.keras")


def calculate_smape_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return SMAPE value in percent using NumPy arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / denominator) * 100
    return smape


def calculate_cv_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of variation of RMSE in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mean_obs = np.mean(y_true)
    if mean_obs == 0:
        return np.nan
    return (rmse / mean_obs) * 100

def main(model_path: str):

    model = tf.keras.models.load_model(model_path)

    batched_predictions = []
    batched_truth = []

    for data in windowed_test:
        (past_inputs, future_inputs), truth = data
        preds = model.predict([past_inputs, future_inputs])
        batched_predictions.append(preds)
        batched_truth.append(truth)

    # Deduplicate by keeping only the last element of each batch
    shifted_predictions = []
    shifted_truth = []
    for i in range(len(batched_predictions) - 1):
        shifted_predictions.append(batched_predictions[i][-1])
        shifted_truth.append(batched_truth[i][-1])
    shifted_predictions.append(batched_predictions[-1][-1])
    shifted_truth.append(batched_truth[-1][-1])

    unique_predictions = [shifted_predictions[0]]
    unique_truth = [shifted_truth[0]]
    for i in range(1, len(shifted_predictions)):
        if not np.array_equal(shifted_predictions[i], unique_predictions[-1]):
            unique_predictions.append(shifted_predictions[i])
        if not np.array_equal(shifted_truth[i], unique_truth[-1]):
            unique_truth.append(shifted_truth[i])

    unique_predictions = np.array(unique_predictions)
    unique_truth = np.array(unique_truth)

    flattened_predictions = unique_predictions.reshape(-1, unique_predictions.shape[-1])
    flattened_truth = unique_truth.reshape(-1, 1)

    # Overall metrics
    mse_overall = np.mean((flattened_truth - flattened_predictions) ** 2)
    mae_overall = np.mean(np.abs(flattened_truth - flattened_predictions))
    rmse_overall = np.sqrt(mse_overall)
    mape_overall = np.mean(np.abs((flattened_truth - flattened_predictions) / flattened_truth)) * 100
    smape_overall = calculate_smape_numpy(flattened_truth, flattened_predictions)
    cv_rmse_overall = calculate_cv_rmse(flattened_truth, flattened_predictions)
    r2 = r2_score(flattened_truth, flattened_predictions)

    # Metrics per 24h sequence
    mse_24 = []
    mae_24 = []
    rmse_24 = []
    mape_24 = []
    smape_24 = []
    cv_rmse_24 = []
    for pr_day, tr_day in zip(unique_predictions.squeeze(), unique_truth.squeeze()):
        mse_d = np.mean((tr_day - pr_day) ** 2)
        mae_d = np.mean(np.abs(tr_day - pr_day))
        rmse_d = np.sqrt(mse_d)
        mape_d = np.mean(np.abs((tr_day - pr_day) / tr_day)) * 100
        smape_d = calculate_smape_numpy(tr_day, pr_day)
        cv_rmse_d = calculate_cv_rmse(tr_day, pr_day)
        mse_24.append(mse_d)
        mae_24.append(mae_d)
        rmse_24.append(rmse_d)
        mape_24.append(mape_d)
        smape_24.append(smape_d)
        cv_rmse_24.append(cv_rmse_d)

    metrics = {
        "mse_per_24h": np.mean(mse_24),
        "mae_per_24h": np.mean(mae_24),
        "rmse_per_24h": np.mean(rmse_24),
        "mape_per_24h": np.mean(mape_24),
        "smape_per_24h": np.mean(smape_24),
        "cv_rmse_per_24h": np.mean(cv_rmse_24),
        "overall_mse": mse_overall,
        "overall_mae": mae_overall,
        "overall_rmse": rmse_overall,
        "overall_mape": mape_overall,
        "overall_smape": smape_overall,
        "overall_cv_rmse": cv_rmse_overall,
        "r2": r2,
    }

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    os.makedirs("Metrics", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join("Metrics", "evaluation_metrics.csv"), index=False)

    fig, ax = plt.subplots()
    ax.plot(flattened_predictions, label="Predictions", c="#ff7f0e")
    ax.plot(flattened_truth, label="True values", c="#2ca02c")
    plt.xlabel("time index")
    plt.ylabel("Demand Normed")
    plt.legend()
    plt.title(f"{model_path} and R^2: {np.round(r2,4)}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict heat demand using a saved model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained Keras model (default: latest in weights/)",
    )
    args = parser.parse_args()
    selected_model = args.model_path or get_latest_model_path()
    main(selected_model)
