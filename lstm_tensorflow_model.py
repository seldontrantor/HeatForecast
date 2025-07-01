"""
Created on Mon Jul 17 13:45:39 2023

@author: AminDar @Github
"""
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sequenced_data import Window_Gen, windowed_train, windowed_test, windowed_val, input_shape, output_shape, BATCH_SIZE
from load_and_norm import load_and_normalize
from utils.lstm_utils import save_model_artifacts, plot_training_history, evaluate_model

sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})

# Load and normalize dataset explicitly for saving artifacts
train_dfs, test_dfs, val_dfs, df_norm_train, df_norm_test, df_norm_val, scaler = load_and_normalize(
    'datasets/df_sin_cosing.csv', columns_to_normalize=['Demand', 'Temp']
)

n_units = 50
learning_rate = 0.0013
loss_fun = 'mean_squared_error'

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=10000,
    decay_rate=0.9
)

def define_models_dense(input_shape, output_shape, n_units=n_units):
    encoder_inputs = Input(shape=input_shape, name='past inputs')
    encoder = LSTM(n_units, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=output_shape, name='future inputs')
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = Dense(64, activation='relu')(decoder_outputs)
    decoder_outputs = Dense(32, activation='relu')(decoder_outputs)
    decoder_outputs = Dense(1, activation='relu')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_outputs = Dense(64, activation='relu')(decoder_outputs)
    decoder_outputs = Dense(32, activation='relu')(decoder_outputs)
    decoder_outputs = Dense(1, activation='relu')(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs, state_h, state_c])

    return model, encoder_model, decoder_model

model, encoder_model, decoder_model = define_models_dense(input_shape, output_shape)

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=lr_schedule),
    loss=loss_fun,
    metrics=['mse', tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()]
)

model.summary()
run_time = datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
os.makedirs(os.path.join("weights", run_time), exist_ok=True)

callbacks_list = [
    TensorBoard(log_dir=os.path.join("logs", run_time), histogram_freq=1, profile_batch='10,30'),
    ModelCheckpoint(os.path.join("weights", run_time,
                                 "weights-improvement-{epoch:02d}-{loss:.2f}.keras"),
                    monitor='loss',
                    verbose=1,
                    save_best_only=False,  # Save after every epoch
                    mode='min' ), # Use 'min' since we want to minimize the loss
    EarlyStopping(monitor='loss', patience=8) # Stop early if loss doesn't improve
]

nEpoch = 3
history = model.fit(windowed_train, epochs=nEpoch, validation_data=windowed_val, verbose=True,
                    shuffle=False, callbacks=callbacks_list)

plot_training_history(history)

save_model_artifacts(
    model=model,
    scaler=scaler,
    history=history,
    output_shape=output_shape,
    batch_size=BATCH_SIZE,
    nEpoch=nEpoch,
    run_time=run_time
)

evaluate_model(model, windowed_test, plot_fn=Window_Gen.plot, take=1)
plt.show()
