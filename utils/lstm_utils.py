import os
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_model_artifacts(model, scaler, history, output_shape, batch_size, nEpoch, run_time, base_path="weights"):
    """Save model weights, whole model, scaler, and training history."""
    os.makedirs(os.path.join(base_path, run_time), exist_ok=True)

    weightsfile = f"weights-completed-bs{batch_size}-epochs{nEpoch}-loss{history.history['loss'][-1]:.4f}-nfeatures{output_shape}.hdf5"
    model.save_weights(os.path.join(base_path, run_time, weightsfile))
    model.save(os.path.join(base_path, run_time, 'whole_model.keras'))

    joblib.dump(scaler, os.path.join(base_path, run_time, "scaler.save"))
    pickle.dump(scaler, open(os.path.join(base_path, run_time, 'scaler.pickle'), 'wb'))

    print(f"Model and scaler saved to {os.path.join(base_path, run_time)}")

def plot_training_history(history):
    """Plot training and validation loss metrics from training history."""
    losses = pd.DataFrame(history.history)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=330)
    sns.lineplot(data=losses, linewidth=3, ax=ax)
    ax.set_title("Training and Validation Metrics")
    ax.set_ylabel("Value")
    ax.set_xlabel("Epoch")
    ax.legend(losses.columns)
    plt.show()

def evaluate_model(model, dataset, plot_fn, take=1):
    """Predict and visualize model outputs on a dataset."""
    for data in dataset.take(take):
        (past, future), truth = data
        predictions = model.predict([past, future])
        plot_fn(past, truth, title=f"Evaluation Take {take}", predictions=predictions)
