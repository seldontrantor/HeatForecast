"""
Created on Mon Jul 10 13:58:01 2023

@author: AminDar @Github
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from load_and_norm import load_and_normalize


class SequencedData:
    def __init__(
        self,
        input_width,
        label_width,
        SHIFT,
        train_df,
        test_df,
        val_df,
        label_columns,
        BATCH_SIZE,
        input_start,
        n_supplementary_features,
        SHUFFLE=False,
    ):
        """
        Class to structure and prepare time series data for encoder-decoder models.
        Splits data into encoder and decoder inputs and provides training/validation/test datasets.
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.SHIFT = SHIFT
        self.BATCH_SIZE = BATCH_SIZE
        self.label_columns = label_columns
        self.input_width = input_width
        self.label_width = label_width
        self.label_columns_indices = {
            name: i for i, name in enumerate(self.label_columns)
        }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        self.total_window_size = input_width + label_width
        self.input_start = input_start
        self.label_start = self.total_window_size - label_width
        self.input_slice = slice(0, input_width)
        self.labels_slice = slice(self.label_start, None)
        self.SHUFFLE = SHUFFLE
        self.buffer_length = self.total_window_size
        self.supplementary_feature = n_supplementary_features
        self.total_features = len(self.column_indices)
        self.deterministic_feature = self.total_features - self.supplementary_feature

    def make_dataset(self, data):
        """Creates a TensorFlow dataset from input data with specified sequence settings."""
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.SHIFT,
            sampling_rate=1,
            shuffle=False,
            batch_size=None,
        )
        ds = ds.batch(self.BATCH_SIZE, drop_remainder=True)
        if self.SHUFFLE:
            ds = ds.shuffle(
                buffer_size=self.buffer_length, seed=42, reshuffle_each_iteration=False
            )
        return ds

    def split_windows_map(self, features):
        """Splits the input sequence into encoder input, decoder input, and decoder output."""
        encoder_inputs = features[:, self.input_slice, :]
        decoder_inputs = features[:, self.labels_slice, self.supplementary_feature :]
        labels_in = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels_in = tf.stack(
                [
                    labels_in[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )
            labels_in = tf.squeeze(labels_in, axis=-1)

        labels_in.set_shape([None, self.label_width])
        encoder_inputs = tf.convert_to_tensor(encoder_inputs)
        encoder_inputs.set_shape([None, self.input_width, None])
        decoder_inputs.set_shape([None, self.label_width, None])
        return encoder_inputs, decoder_inputs, labels_in

    def transform_element(self, x, y, z):
        """Groups encoder and decoder inputs as one tuple, and decoder labels as target."""
        return (x, y), z

    def train(self):
        """Returns the training dataset."""
        return (
            self.make_dataset(self.train_df)
            .map(self.split_windows_map)
            .map(self.transform_element)
            .prefetch(tf.data.AUTOTUNE)
        )

    def val(self):
        """Returns the validation dataset."""
        return (
            self.make_dataset(self.val_df)
            .map(self.split_windows_map)
            .map(self.transform_element)
            .prefetch(tf.data.AUTOTUNE)
        )

    def test(self):
        """Returns the test dataset."""
        self.SHUFFLE = False
        return (
            self.make_dataset(self.test_df)
            .map(self.split_windows_map)
            .map(self.transform_element)
            .prefetch(tf.data.AUTOTUNE)
        )

    def plot(
        self, inputs, labels, title, predictions=None, plot_col="Demand", max_subplots=3
    ):
        """Visualizes input, true labels, and optionally predictions for inspection."""
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                np.arange(self.input_width)[self.input_slice],
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )
            label_col_index = self.label_columns_indices.get(plot_col)
            plt.scatter(
                np.arange(self.total_window_size)[self.labels_slice],
                labels[n, :],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )
            if predictions is not None:
                plt.scatter(
                    np.arange(self.total_window_size)[self.labels_slice],
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )
            if n == 0:
                plt.legend()
                plt.title(f" Predictions for the take number: {title}")
        plt.xlabel("Time [h]")


def load_default_data():
    """Return preconfigured ``SequencedData`` and windowed datasets."""
    train_dfs, test_dfs, val_dfs, df_norm_train, df_norm_test, df_norm_val, scaler = (
        load_and_normalize(
            "datasets/df_sin_cosing.csv", columns_to_normalize=["Demand", "Temp"]
        )
    )

    BATCH_SIZE = 32
    Window_Gen = SequencedData(
        input_width=24,
        label_width=24,
        SHIFT=1,
        train_df=df_norm_train,
        test_df=df_norm_test,
        val_df=df_norm_val,
        label_columns=["Demand"],
        BATCH_SIZE=BATCH_SIZE,
        input_start=1,
        n_supplementary_features=len(["Demand"]),
        SHUFFLE=True,
    )

    windowed_train = Window_Gen.train()
    windowed_val = Window_Gen.val()
    windowed_test = Window_Gen.test()
    input_len = Window_Gen.input_width
    forecast_len = Window_Gen.label_width
    input_shape = windowed_train.element_spec[0][0].shape[1:]
    output_shape = windowed_train.element_spec[0][1].shape[1:]

    return {
        "Window_Gen": Window_Gen,
        "windowed_train": windowed_train,
        "windowed_val": windowed_val,
        "windowed_test": windowed_test,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "BATCH_SIZE": BATCH_SIZE,
        "input_len": input_len,
        "forecast_len": forecast_len,
        "df_norm_train": df_norm_train,
        "df_norm_test": df_norm_test,
        "df_norm_val": df_norm_val,
        "scaler": scaler,
    }


if __name__ == "__main__":
    data = load_default_data()
    Window_Gen = data["Window_Gen"]
    print("Input length: ", data["input_len"])
    print("Forecast length: ", data["forecast_len"])
    print("Batch size: ", data["BATCH_SIZE"])
    print("Total features: ", Window_Gen.total_features)
    print("Determanistic features: ", Window_Gen.determanistic_feature)
    print("Label columns: ", Window_Gen.label_columns)
    print("Label columns indices: ", Window_Gen.label_columns_indices)
    print("Column indices: ", Window_Gen.column_indices)
    print("Total window size: ", Window_Gen.total_window_size)
