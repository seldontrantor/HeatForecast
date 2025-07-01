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
from sequenced_data import load_default_data
from utils.lstm_utils import save_model_artifacts, plot_training_history, evaluate_model
import argparse

sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})

n_units = 50
loss_fun = 'mean_squared_error'

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


def main(nepoch,learning_rate):

    data = load_default_data()
    Window_Gen = data["Window_Gen"]
    windowed_train = data["windowed_train"]
    windowed_val = data["windowed_val"]
    windowed_test = data["windowed_test"]
    input_shape = data["input_shape"]
    output_shape = data["output_shape"]
    BATCH_SIZE = data["BATCH_SIZE"]
    scaler = data["scaler"]

    print(f"Training model with {nepoch} epochs and learning rate of {learning_rate}")
    print(f"Training model with input shape {input_shape} and output shape of {output_shape}")
    print(f"Training model with batch size of {BATCH_SIZE}")

    model, encoder_model, decoder_model = define_models_dense(input_shape, output_shape)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        decay_rate=0.9
    )

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
        ModelCheckpoint(os.path.join("weights", run_time, "weights-improvement-{epoch:02d}-{loss:.2f}.keras"), monitor='loss', verbose=1, save_best_only=False, mode='max'),
        EarlyStopping(monitor='loss', patience=8)
    ]

    nEpoch = nepoch
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a LSTM model on the default dataset"
    )
    parser.add_argument(
        "--nepoch",
        type=int,
        default=250,
        help="Number of epochs to train the model (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0013,
        help="Learning rate for the optimizer (default: 0.0013)",
    )
    args = parser.parse_args()
    nepoch = args.nepoch
    learning_rate = args.learning_rate
    main(nepoch, learning_rate)
