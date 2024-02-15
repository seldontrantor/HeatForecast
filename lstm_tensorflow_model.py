"""
Created on Mon Jul 17 13:45:39 2023

@author: Amin Darbandi
"""
import datetime

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input
from my_s2s_widnow import Window_Gen, windowed_train, windowed_test, windowed_val, input_shape, output_shape, BATCH_SIZE, \
    train_dfs, test_dfs, val_dfs, scaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from sklearn.metrics import r2_score
import pickle
import joblib


sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})

# returns train, inference_encoder and inference_decoder models

n_units = 32
# Optimizer
learning_rate = 0.0013
loss_fun ='mean_squared_error'

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay (
    initial_learning_rate=learning_rate,
    decay_steps=10000,
    decay_rate=0.9)


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


# Assuming you have input observations and target labels as tensors

# Define the shapes
in_shape = input_shape  # Shape of input observations tensor
out_shape = output_shape  # Shape of target labels tensor

# Create the model

DENSE = True

if DENSE is not False:
    model, encoder_model, decoder_model = define_models_dense(input_shape, output_shape)
else:
    model, encoder_model, decoder_model = define_models(input_shape, output_shape)


# Number of episodes
nEpoch = 250

optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
# Compile the model
model.compile(optimizer=optimizer,
              loss= loss_fun,
              metrics=['mse',
                       tf.keras.metrics.MeanAbsoluteError(), 
                       tf.keras.metrics.RootMeanSquaredError(),
                       tf.keras.metrics.MeanAbsolutePercentageError()])

# Model Summery
model.summary()
run_time = datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
os.makedirs("weights" + os.sep + run_time, exist_ok=True)

# Create a TensorBoard instance with the path to the logs directory
tensorboard = TensorBoard(log_dir='logs/{}'.format(run_time))

# Checkpoint
filepath = os.path.join("weights", run_time, "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='max')
checkpoint_1 = EarlyStopping(monitor='loss', patience=5)

callbacks_list = [tf.keras.callbacks.TensorBoard(
                                                os.path.join("logs", run_time),
                                                histogram_freq=1,profile_batch = '10,30'),
                                                 checkpoint,
                                                 checkpoint_1]

# Train the model
history = model.fit(windowed_train, epochs=nEpoch, validation_data=windowed_val, verbose=True,
                    shuffle=False, callbacks=callbacks_list)

weightsfile = "weights-completed-bs{}-epochs{}-loss{}-nfeatures{}.hdf5".format(
    BATCH_SIZE,
    nEpoch,
    '{:.4f}'.format(history.history['loss'][-1]),
    output_shape)
model.save_weights(os.path.join("weights", run_time, weightsfile))
model.save(os.path.join("weights", run_time, 'whole_model.keras'))

joblib.dump(scaler, os.path.join("weights",run_time,"scaler.save"))
pickle.dump(scaler, open(os.path.join("weights",run_time,'scaler.pickle'),'wb'))

losses = pd.DataFrame(history.history)
fig, ax = plt.subplots(figsize=(16, 8), dpi=330)
ax = sns.lineplot(data=losses, linewidth=3)


take = 1
# Loop through the 'windowed_test' dataset to get true and predicted values
for data in windowed_test.take(take):
    # Predict using the model
    (past, future), truth = data
    
    truth = truth
    predictions = model.predict([past, future])

inputs = past
labels = truth
    
Window_Gen.plot(inputs, labels,take, predictions)

