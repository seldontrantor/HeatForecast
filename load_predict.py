import datetime
from sklearn.metrics import r2_score
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input
from my_s2s_widnow import windowed_train, windowed_test, windowed_val, input_shape, output_shape, BATCH_SIZE, \
     train_dfs, test_dfs, val_dfs,df_norm_test
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})

n_units = 32


# model Dense

def define_models_dense(input_shape, output_shape, n_units=n_units):
    """
    Define the encoder-decoder LSTM model.

    :param input_shape: shape of observation inputs
    :param output_shape: shape of output prediction length
    :param n_units:  The number of cells to create in the encoder and decoder models,
    :return: model as encoder_input, decoder_input, decoder_output as tuple
            and encoder decoder model
    """

    # Encoder
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=output_shape)
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense1 = Dense(64, activation='relu')  # Add additional dense layers here
    decoder_dense2 = Dense(32, activation='relu')  # Add more layers if needed
    decoder_dense_output = Dense(1, activation='relu')  # Output layer
    decoder_outputs = decoder_dense1(decoder_outputs)
    decoder_outputs = decoder_dense2(decoder_outputs)
    decoder_outputs = decoder_dense_output(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Define the encoder model
    encoder_model = Model(encoder_inputs, encoder_states)

    # Define the decoder model
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense1(decoder_outputs)
    decoder_outputs = decoder_dense2(decoder_outputs)
    decoder_outputs = decoder_dense_output(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def define_models(input_shape, output_shape, n_units=n_units):
    """
    Define the encoder-decoder LSTM model.

    :param input_shape: shape of observation inputs
    :param output_shape: shape of output prediction length
    :param n_units:  The number of cells to create in the encoder and decoder models,
    :return: model as encoder_input, decoder_input, decoder_output as tuple
            and encoder decoder model
    """

    # Encoder
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=output_shape)
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(16, activation='relu')  # Only one dense layer
    decoder_dense_output = Dense(1, activation='relu')  # Output layer
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_outputs = decoder_dense_output(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Define the encoder model
    encoder_model = Model(encoder_inputs, encoder_states)

    # Define the decoder model
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_outputs = decoder_dense_output(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def load_trained_model(path, input_shape, output_shape):
    """
    Load a trained model into a Keras model object to use for prediction

    Arguments
    ---------
    weights_path : str
        Path to weights file (.h5) on system
    input_shape : int
        Number of time steps as input to algorithm
    output_shape : int
        Number of time steps in forecast/output/prediction
    n_units : int
        Number of input features used for prediction

    Returns
    -------
    model : object (Keras model)
        Keras model object loaded with weights from training experiment
    """
    model, encoder_model, decoder_model = define_models_dense(input_shape, output_shape)
    model.load_weights(path)
    return model


path = input('input the model path:')
path = path.strip("'")
model_weight = load_trained_model(path, input_shape, output_shape)

model_whole = tf.keras.models.load_model(path)

whole = True
if whole:
    model = model_whole
else:
    model = model_weight
    
    
df_norm = df_norm_test.copy()
df_norm = df_norm.astype('float32')



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

fig, ax = plt.subplots()
ax.plot(all_predictions, label='Predictions',c='#ff7f0e')
ax.plot(all_truth,label='True values', c='#2ca02c')
plt.xlabel('time index')
plt.ylabel('Demand Normed')
plt.legend()
plt.title(f'{path[-38:-18]} and R^2: {np.round(r2,4)}')






def define_models(input_shape, output_shape, n_units=n_units):
    """
    Define the encoder-decoder LSTM model.

    :param input_shape: shape of observation inputs
    :param output_shape: shape of output prediction length
    :param n_units:  The number of cells to create in the encoder and decoder models,
    :return: model as encoder_input, decoder_input, decoder_output as tuple
            and encoder decoder model
    """

    # Encoder
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=output_shape)
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = tf.keras.layers.Dense(16, activation='relu')(decoder_outputs)
    decoder_outputs = tf.keras.layers.Dense(16, activation='relu')(decoder_outputs)
    decoder_outputs = tf.keras.layers.Dense(1, activation='relu')(decoder_outputs)


    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Define the encoder model
    encoder_model = Model(encoder_inputs, encoder_states)

    # Define the decoder model
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model



