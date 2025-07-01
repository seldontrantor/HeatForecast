import os
import argparse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sequenced_data import load_default_data



sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})



def get_latest_model_path(weights_dir: str = "weights") -> str:
    """Return path to the latest saved model inside ``weights_dir``.

    Parameters
    ----------
    weights_dir : str
        Directory containing timestamped subfolders with ``whole_model.keras``.

    Returns
    -------
    str
        Absolute path to the most recently modified ``whole_model.keras`` file.
    """
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"No weights directory found at {weights_dir}")

    subdirs = [
        os.path.join(weights_dir, d)
        for d in os.listdir(weights_dir)
        if os.path.isdir(os.path.join(weights_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No model directories found in {weights_dir}")

    latest_dir = max(subdirs, key=os.path.getmtime)
    candidate = os.path.join(latest_dir, "whole_model.keras")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Model not found at {candidate}")
    return candidate


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
    data = load_default_data()
    input_len = data["input_len"]
    forecast_len = data["forecast_len"]
    df_test = data["df_norm_test"]

    model = tf.keras.models.load_model(model_path)

    all_predictions = []
    all_truth = []
    metrics_per_24h = []
    test_len = len(df_test)

    for start in range(0, test_len - input_len - forecast_len + 1, forecast_len):
        end = start + input_len
        next_end = end + forecast_len

        # Extract the past window_len data points
        past_window = df_test[start:end]
        future_window = df_test[end:next_end]

        # Reshape the data to fit the model's input shape
        past_window_input = past_window.values.reshape((1, input_len, past_window.shape[1]))

        # Extract future deterministic features for the forecast period
        future_inputs = future_window.iloc[:, 1:]
        future_inputs = future_inputs.values.reshape((1, forecast_len, future_inputs.shape[1]))

        # Make predictions
        predictions = model.predict([past_window_input, future_inputs])

        # Flatten predictions and store them
        predictions = predictions.flatten()
        all_predictions.extend(predictions)
        truth = future_window['Demand'].values.flatten()  # Assuming 'Demand' is the target variable
        all_truth.extend(truth)

        # Calculate metrics for this 24-hour prediction set
        mse = mean_squared_error(truth, predictions)
        mae = mean_absolute_error(truth, predictions)
        r2 = r2_score(truth, predictions)
        mape = mean_absolute_percentage_error(truth, predictions)
        smape = calculate_smape_numpy(truth, predictions)
        cv_rmse = calculate_cv_rmse(truth, predictions)
        rsmess = np.sqrt(np.mean((truth - predictions) ** 2))

        metrics_per_24h.append(
            {
            'start': start,
            'end': next_end,
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'SMAPE': smape,
            'CV-RSME': cv_rmse,
            'RSME': rsmess
            }
            )

    # Convert predictions and truth values to numpy arrays for evaluation
    all_predictions = np.array(all_predictions)
    all_truth = np.array(all_truth)
    # Overall metrics
    overall_mse = mean_squared_error(all_truth, all_predictions)
    overall_mae = mean_absolute_error(all_truth, all_predictions)
    overall_r2 = r2_score(all_truth, all_predictions)
    overall_mape = mean_absolute_percentage_error(all_truth, all_predictions)
    overall_smape = calculate_smape_numpy(all_truth, all_predictions)
    overall_cv_rsme = calculate_cv_rmse(all_truth, all_predictions)
    overall_rsme = np.sqrt(np.mean((all_truth - all_predictions) ** 2))

    metrics_overall = {
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "overall_rmse":  overall_rsme,
        "overall_mape": overall_mape,
        "overall_smape": overall_smape,
        "overall_cv_rmse": overall_cv_rsme,
        "r2": overall_r2
    }

    # Convert metrics_per_24h to DataFrame for easier analysis and visualization
    df_metrics_24 = pd.DataFrame(metrics_per_24h)
    overall_metrics = pd.DataFrame(metrics_overall, index=[0])

    metric_file = model_path.split('/')[1]
    metric_file = metric_file.strip().replace(" ", "_")

    os.makedirs("Metrics", exist_ok=True)
    df_metrics_24.to_csv(os.path.join("Metrics", f"evaluation_metrics_per24_{metric_file}.csv"), index=True)
    overall_metrics.to_csv(os.path.join("Metrics", f'overall_metrics_{metric_file}.csv'), index=True)

    df_metrics_24.plot(x='start', y=['MSE', 'MAE', 'R2'], subplots=True, figsize=(12, 8),
                    title='Evaluation Metrics for Each 24-Hour Prediction Set')
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
