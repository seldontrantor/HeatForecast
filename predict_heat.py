from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sequenced_data import  windowed_train, windowed_test, windowed_val


sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})

model_path = "weights/30-Jun-2025-15-37-15/whole_model.keras"

def main():

    model = tf.keras.models.load_model(model_path)
    all_predictions = []
    all_truth = []
    # Iterate through all takes in the test set
    for data in windowed_test:
        (past_inputs, future_inputs), truth = data
        
        # Make predictions using the model
        predictions = model.predict([past_inputs, future_inputs])
        
        # Convert to NumPy arrays and append to lists
        all_predictions.append(predictions[0])
        all_truth.append(np.array(truth[0]))
    
    # Convert lists to NumPy arrays for easier manipulation
    all_predictions = np.array(all_predictions)
    all_truth = np.array(all_truth)
    
    # Flatten the arrays to have shape (num_samples, num_time_steps)
    flattened_predictions = all_predictions.reshape(-1, all_predictions.shape[-1])
    flattened_truth = all_truth.reshape(-1,1)
    np.savetxt(f"Metrics/{model_path[8:-18]}_flattened_truth.csv", flattened_truth, delimiter=',')
    np.savetxt( f"Metrics/{model_path[8:-18]}_flattened_predictions.csv", flattened_predictions, delimiter=',')
    # Calculate R^2 score
    r2 = r2_score(flattened_truth, flattened_predictions)
    print(f'R^2 Score for the entire test set: {r2}')

    predictions = []
    true_val = []
    # Iterate over the test set and make predictions
    for i, sequence in enumerate(windowed_test):
        (past, future), truth = sequence
        prediction = model.predict([past, future])
        predictions.append(prediction)
        true_val.append(truth)
    
    # Shift predictions to handle duplicates caused by sequential data
    shifted_predictions = []
    shifted_truth = []
    # Iterate over the predictions
    for i in range(len(predictions) - 1):
        # Append the last prediction of each sequence
        shifted_predictions.append(predictions[i][-1])
        shifted_truth.append(true_val[i][-1])
    
    # Append the last prediction of the last sequence
    shifted_predictions.append(predictions[-1][-1])
    shifted_truth.append(true_val[-1][-1])
    # Overwrite duplicated values
    unique_predictions = [shifted_predictions[0]]
    unique_true = [shifted_truth[0]]

    for i in range(1, len(shifted_predictions)):
        # Compare arrays element-wise
        if not np.array_equal(shifted_predictions[i], unique_predictions[-1]):
            unique_predictions.append(shifted_predictions[i])
            
    for i in range(1, len(shifted_predictions)):
        # Compare arrays element-wise
        if not np.array_equal(shifted_truth[i], unique_true[-1]):
            unique_true.append(shifted_truth[i])
    
    
    all_predictions = np.concatenate(unique_predictions, axis=0)
    all_truth = np.concatenate(unique_true, axis=0)
    np.savetxt('s.csv',all_predictions,  delimiter=',')
    np.savetxt('t.csv', all_truth, delimiter=',')
    fig, ax = plt.subplots()
    ax.plot(all_predictions, label='Predictions',c='#ff7f0e')
    ax.plot(all_truth,label='True values', c='#2ca02c')
    plt.xlabel('time index')
    plt.ylabel('Demand Normed')
    plt.legend()
    plt.title(f"{model_path} and R^2: {np.round(r2,4)}")
    plt.show()

if __name__ == "__main__":
    main()
