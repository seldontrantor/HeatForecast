#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:53:22 2024

@author: amin
"""

import tensorflow as tf
import datetime
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
import pickle
import joblib

sns.set()
sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})

df = pd.read_csv('df_sin_cosing.csv', parse_dates=[0, 15], index_col=0)
df = df.drop(['date', 'weekday'], axis=1)
df = df.astype('float')

random_split = False
future_importance = True

if future_importance is True:
    # df = df.drop(['Holidays', 'sin_month', 'cos_month', 'quarter',
    #               'sin_year', 'cos_year'], axis=1)
    
    df = df.drop(['quarter', 
           'sin_dow', 'cos_dow', 'sin_dom',
           'cos_dom','sin_year', 'cos_year','solar',
           'Holidays',
           'wind', 'humid',
           'sin_doy', 'cos_doy', 'sin_month', 'cos_month', 'sin_woy','cos_woy'
           ], axis=1)
    
    
else:
    pass


train_dfs = df.iloc[:8760, :]
val_dfs = df.iloc[8760: 8760 * 2, :]
test_dfs = df.iloc[8760 * 2:, :]

normalized_x = ['Demand', 'Temp']



# ss = StandardScaler()
mm = MinMaxScaler()
# rob = RobustScaler()


def norm_data(train_df, test_df, val_df, scaler, normalize_all_features=False, columns_to_normalize=None):
    """
    Normalize specified columns in the input dataframes using the given scaler.

    Parameters:
        train_df: training dataframes
        test_df: testing dataframes
        val_df: validation dataframes
        scaler: An instance of a scaler (e.g., StandardScaler, MinMaxScaler).
        normalize_all_features (bool): If True, normalize all columns. If False, normalize only specified columns.
        columns_to_normalize (list): List of column names to normalize (used if normalize_all_features is False).

    Returns:
        tuple: A tuple of normalized dataframes df_norm_train, df_norm_test, df_norm_val and the scaler.
    """

    df_norm_train = train_df.copy()
    df_norm_test = test_df.copy()
    df_norm_val = val_df.copy()

    if normalize_all_features:
        df_norm_train[df_norm_train.columns] = scaler.fit_transform(df_norm_train[df_norm_train.columns])
        df_norm_test[df_norm_train.columns] = scaler.transform(df_norm_test[df_norm_train.columns])
        df_norm_val[df_norm_train.columns] = scaler.transform(df_norm_val[df_norm_train.columns])
    
    if not normalize_all_features:
        df_norm_train[columns_to_normalize] = scaler.fit_transform(df_norm_train[columns_to_normalize])
        df_norm_test[columns_to_normalize] = scaler.transform(df_norm_test[columns_to_normalize])
        df_norm_val[columns_to_normalize] = scaler.transform(df_norm_val[columns_to_normalize])

    return df_norm_train, df_norm_test, df_norm_val, scaler


df_norm_train, df_norm_test, df_norm_val, scaler = norm_data(
    train_dfs,
    test_dfs,
    val_dfs,
    mm,
    normalize_all_features = False,
    columns_to_normalize=normalized_x)


def create_dataset(df, n_deterministic_features,
                   window_size, forecast_size,
                   batch_size):
    # Feel free to play with shuffle buffer size
    shuffle_buffer_size = len(df)
    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    data = tf.data.Dataset.from_tensor_slices(df.values)

    # Selecting windows
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    # Shuffling data (seed=Answer to the Ultimate Question of Life, the Universe, and Everything)
    data = data.shuffle(shuffle_buffer_size, seed=42)

    # Extracting past features + deterministic future + labels
    data = data.map(lambda k: ((k[:-forecast_size],
                                k[-forecast_size:, -n_deterministic_features:]),
                               k[-forecast_size:, 0]))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# How much data from the past should we need for a forecast?
window_len = 2 * 7 * 24  # two weeks
# How far ahead do we want to generate forecasts?
forecast_len = 1*24    # Five days



# additional constants
n_total_features = len(df_norm_train.columns)
n_aleatoric_features = len(['Demand'])
n_deterministic_features = n_total_features - n_aleatoric_features

# Splitting dataset into train/val/test
training_data = df_norm_train
validation_data = df_norm_val
test_data = df_norm_test

BATCH_SIZE = 32

training_windowed = create_dataset(training_data,
                                   n_deterministic_features,
                                   window_len,
                                   forecast_len,
                                   BATCH_SIZE)

validation_windowed = create_dataset(validation_data,
                                     n_deterministic_features,
                                     window_len,
                                     forecast_len,
                                     BATCH_SIZE)

test_windowed = create_dataset(test_data,
                               n_deterministic_features,
                               window_len,
                               forecast_len,
                               batch_size=1)



for i in training_windowed.take(1):
    (enc, dec_in), dec_out = i

print('All shapes are: (batch, time, features)')
print(f'Encoder inputs shape: {enc.shape}')
print(f'Decoder inputs shape: {dec_in.shape}')
print(f'Decoder outputs shape: {dec_out.shape}')
input_shape = enc.shape[1:]  # Shape of input observations tensor
output_shape = dec_in.shape[1:]  # Shape of target labels tensor


latent_dim = 32

# First branch of the net is an lstm which finds an embedding for the past
past_inputs = tf.keras.Input(
    shape=(window_len, n_total_features), name='past_inputs')
# Encoding the past
encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(past_inputs)

future_inputs = tf.keras.Input(
    shape=(forecast_len, n_deterministic_features), name='future_inputs')
# Combining future inputs with recurrent branch output
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
x = decoder_lstm(future_inputs,
                 initial_state=[state_h, state_c])

x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='relu')(x)

model = tf.keras.models.Model(
    inputs=[past_inputs, future_inputs], outputs=output)



nEpoch = 250
learning_rate = 0.0013
loss_fun ='mean_squared_error'

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay (
    initial_learning_rate=learning_rate,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss= loss_fun,
              metrics=['mse',
                       tf.keras.metrics.MeanAbsoluteError(), 
                       tf.keras.metrics.RootMeanSquaredError(),
                       tf.keras.metrics.MeanAbsolutePercentageError()])


# Model Summery
model.summary()
run_time = datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
os.makedirs("weights_for_simple_model" + os.sep + run_time, exist_ok=True)

# Create a TensorBoard instance with the path to the logs directory
tensorboard = TensorBoard(log_dir='logs/{}'.format(run_time))

# Checkpoint
filepath = os.path.join("weights_for_simple_model", run_time, "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='max')
checkpoint_1 = EarlyStopping(monitor='loss', patience=5)

callbacks_list = [tf.keras.callbacks.TensorBoard(
                                                os.path.join("logs",run_time),
                                                histogram_freq=1,
                                                profile_batch = '10,30'),
                                                 checkpoint,
                                                 checkpoint_1]

# Train the model
history = model.fit(training_windowed, epochs=nEpoch, validation_data=validation_windowed, 
                    verbose=True,
                  callbacks=callbacks_list)

weightsfile = "weights-completed-bs{}-epochs{}-loss{}-nfeatures{}.hdf5".format(
    BATCH_SIZE,
    nEpoch,
    '{:.4f}'.format(history.history['loss'][-1]),
    output_shape)
model.save_weights(os.path.join("weights_for_simple_model", run_time, weightsfile))
model.save(os.path.join("weights_for_simple_model", run_time, 'whole_model.keras'))

joblib.dump(scaler, os.path.join("weights_for_simple_model",run_time,"scaler.save"))
pickle.dump(scaler, open(os.path.join("weights_for_simple_model",run_time,'scaler.pickle'),'wb'))

losses = pd.DataFrame(history.history)
fig, ax = plt.subplots(figsize=(16, 8), dpi=330)
ax = sns.lineplot(data=losses, linewidth=3)






for data in test_windowed.take(1):
  (past, future), truth = data

  truth = truth
  pred = model.predict((past,future))
  
inputs = past
labels = truth

predictions = pred
max_n = 3  

# if the forcast horizon is 1h
for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel('demand')
    plt.plot(np.arange(7)[slice(0, 7)], inputs[n, :,0],
              label='Inputs', marker='.', zorder=-10)
    plt.scatter(np.arange(10)[slice(7, 8)], labels[n, :, ][0],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    predictions = pred
    plt.scatter(np.arange(10)[slice(7, 8)], predictions[n, :, 0],
                marker='X', edgecolors='k', label='Predictions',
                c='#ff7f0e', s=64)
    if n == 0:
      plt.legend()

plt.xlabel('Time [h]')


# if the wforecast horizon is > 1
take  = 10
for data in test_windowed.take(take):
  (past, future), truth = data

  truth = truth
  pred = model.predict((past,future))

inputs = past
labels = truth

predictions = pred
max_n = 0  

for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel('Demand[Normalized]')
    
    plt.plot(np.arange(window_len)[slice(0, window_len)], inputs[n, :,0],
             label='Inputs', marker='.', zorder=-10)
    plt.scatter(np.arange(window_len+forecast_len)[slice(window_len, window_len+forecast_len)], 
                labels[n, :, ],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    predictions = pred
    plt.scatter(np.arange(window_len+forecast_len)[slice(window_len, window_len+forecast_len)], predictions[n, :, 0],
                marker='X', edgecolors='k', label='Predictions',
                c='#ff7f0e', s=64)
    plt.xlabel('Time Step')
    if n == 0:
      
      plt.legend()
      plt.title(f'Samples from take {take}')
      
      
      

# Assuming 'test_windowed' is your test data generator or dataset
# Assuming 'model' is your trained model

# Lists to store predictions and truth values
all_predictions = []
all_truth = []

# Iterate through all takes in the test set
for data in test_windowed:
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
      
from keras.utils.vis_utils import plot_model
import pydot
import pydotplus
from keras.utils.vis_utils import model_to_dot
import keras
keras.utils.vis_utils.pydot = pydot
plot_model(model, to_file='model.png')


fig, ax = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all')

# We need to rescale cnt
scaling_factor = 1

for i, data in enumerate(test_windowed.take(6)):
  (past, future), truth = data

  truth = truth * scaling_factor
  pred = model.predict((past,future)) * scaling_factor

  row = i//2
  col = i%2

  ax[row][col].plot(pred.flatten(), label='Prediction')
  ax[row][col].plot(truth.numpy().flatten(),label='Truth')

# Labeling axes
for i in range(2):
  ax[2][i].set_xlabel('Hour interval')
for i in range(3):
  ax[i][0].set_ylabel('Used bikes')

handles, labels = ax[0][0].get_legend_handles_labels()
fig.subplots_adjust(wspace=0, hspace=0.5)
fig.legend(handles, labels, loc='upper right')