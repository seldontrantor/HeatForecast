"""
Created on Mon Jul 10 13:58:01 2023

@author: Amin Darbandi
"""
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score
sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})

df = pd.read_csv('df_sin_cosing.csv', parse_dates=[0, 15], index_col=0)
df = df.drop(['date', 'weekday'], axis=1)
df = df.astype('float64')


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



def split_data(dataframe, split=0.8):
    """

  split time series into train/test sets
  : param df:                      data frame
  : para split:                   percent of data to include in training set
  : return t_train, y_train:      time/feature training and test sets;
  :        t_test, y_test:        (shape: [# samples, 1])

  """

    n = len(dataframe)
    split_valid = split + 0.1

    train_dfs = dataframe[0:int(n * split)]
    val_dfs = dataframe[int(n * split): int(n * split_valid)]
    test_dfs = dataframe[int(n * split_valid):]

    return train_dfs, test_dfs, val_dfs


if random_split is False:
    train_dfs = df.iloc[:8760, :]
    val_dfs = df.iloc[8760: 8760 * 2, :]
    test_dfs = df.iloc[8760 * 2:, :]


else:
    train_dfs, test_dfs, val_dfs = split_data(df)

# normalized_x = ['Demand', 'Temp', 'solar', 'wind', 'humid']
normalized_x = ['Demand','Temp']

# ss = StandardScaler()
mm = MinMaxScaler()
# rob = RobustScaler()


def norm_data(train_df, test_df, val_df, scaler, normalize_all_features=True, columns_to_normalize=None):
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


class SequencedData():

    def __init__(self, input_width,
                 label_width,
                 SHIFT,
                 train_df,
                 test_df,
                 val_df,
                 label_columns,
                 BATCH_SIZE,
                 input_start,
                 n_supplementary_features, 
                 SHUFFLE=False):
        """
        Initialize the SequencedData object.

        This class is used to preprocess and organize sequential data for training
        machine learning models.

        Parameters:
            input_width (int): The size of the input window (sequence) for the model past values.
            label_width (int): The size of the label window (output) for the model. Forecast horizon
            SHIFT (int): The number of time steps to shift the label window.
            train_df (DataFrame): The training dataset containing the input and label data.
            test_df (DataFrame): The test dataset containing the input and label data.
            val_df (DataFrame): The validation dataset containing the input and label data.
            label_columns (list): List of column names representing the target labels.
            BATCH_SIZE (int): The batch size used during training and evaluation.
            input_start (int): The starting index for the input window in each sequence.
            SHUFFLE (bool, optional): Whether to shuffle the data during training. Default is False.

        Note:
            The DataFrame variables (train_df, test_df, val_df) must contain columns for
            both input and label data, as specified by input_width and label_width.

        Examples:
            # Create a SequencedData instance
            w1 = SequencedData(
                input_width=24,
                label_width=2,
                SHIFT=2,
                train_df=train_df,
                test_df=test_df,
                val_df=val_df,
                label_columns=['Demand'],
                BATCH_SIZE=128,
                input_start=1,
                SHUFFLE=False
            )
        """

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.SHIFT = SHIFT
        self.BATCH_SIZE = BATCH_SIZE
        self.label_columns = label_columns
        self.input_width = input_width
        self.label_width = label_width
        self.label_columns_indices = {name: i for i, name in
                                      enumerate(self.label_columns)}

        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        self.total_window_size = input_width + self.label_width
        self.input_start = input_start
        self.label_start = self.total_window_size - self.label_width
        self.input_slice = slice(0, input_width)
        self.labels_slice = slice(self.label_start, None)
        self.SHUFFLE = SHUFFLE
        self.buffer_length = self.input_width + self.label_width
        self.supplementary_feature = n_supplementary_features
        self.total_features = len(self.column_indices)
        self.determanistic_feature = self.total_features - self.supplementary_feature

    def make_dataset(self, data, ):
        """
               Create a time series dataset from the input data.

               Parameters:
                   data (numpy.ndarray or pandas.DataFrame): The input data containing sequences.

               Returns:
                   tf.data.Dataset: A TensorFlow Dataset containing the time series data.

               Note:
                   This method is used internally to organize and transform the data into
                   the required format for training and evaluation.

               Examples:
                   # Create a time series dataset
                   dataset = Window_Gen.make_dataset(train_df)
               """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.SHIFT,
            sampling_rate=1,
            shuffle=False,
            batch_size=None)

        if not self.SHUFFLE:

            ds = ds.batch(self.BATCH_SIZE, drop_remainder=True)

        else:

            ds = ds.batch(
                self.BATCH_SIZE,
                drop_remainder=True).shuffle(
                buffer_size=self.buffer_length,
                seed=42,
                reshuffle_each_iteration=False
            )

        return ds

    def split_windows_map(self, features):
        """
                Split the time series data into input and label windows.

                Parameters:
                    features (tf.Tensor): The features tensor containing the input and label data.

                Returns:
                    tuple: A tuple of (input_sequence, decoder_input, decoder_output) tensors.

                Note:
                    This method is used internally by the `make_dataset` method.

                Examples:
                    # Split the time series data into input and label windows
                    input_sequence, decoder_input, decoder_output = Window_Gen.split_windows_map(features)
                """

        encoder_inputs = features[:, self.input_slice, :]
        decoder_inputs = features[:, self.labels_slice, self.supplementary_feature:]
        labels_in = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels_in = tf.stack(
                [labels_in[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
            labels_in = tf.squeeze(labels_in, axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.

        labels_in.set_shape([None, self.label_width])

        encoder_inputs = tf.convert_to_tensor(encoder_inputs)
        encoder_inputs.set_shape([None, self.input_width, None])
        decoder_inputs.set_shape([None, self.label_width, None])

        # labels_in.set_shape([None, self.label_width, None])

        return encoder_inputs, decoder_inputs, labels_in

    def transform_element(self, x, y, z):
        # Combine the first two tensors into a tuple
        xy_tuple = (x, y)
        # Create a tuple with the resulting tuple and the third tensor
        return xy_tuple, z

    def train(self):
        """
        Get the training dataset for the SequencedData object.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset containing the training data.

        Note:
            This method returns the training dataset, which is suitable for use in training
            a machine learning model.

        Examples:
            # Get the training dataset
            train_dataset = Window_Gen.train()
        """
        trained_window = self.make_dataset(self.train_df)
        trained_window = trained_window.map(self.split_windows_map)
        trained_window = trained_window.map(self.transform_element)

        return trained_window.prefetch(tf.data.experimental.AUTOTUNE)

    def val(self):
        """
                Get the validation dataset for the SequencedData object.

                Returns:
                    tf.data.Dataset: A TensorFlow Dataset containing the validation data.

                Note:
                    This method returns the validation dataset, which is suitable for use in
                    evaluating the performance of a machine learning model during training.

                Examples:
                    # Get the validation dataset
                    val_dataset = Window_Gen.val()
                """
        val_window = self.make_dataset(self.val_df)
        val_window = val_window.map(self.split_windows_map)
        val_window = val_window.map(self.transform_element)

        return val_window.prefetch(tf.data.experimental.AUTOTUNE)

    def test(self):
        """
               Get the test dataset for the SequencedData object.

               Returns:
                   tf.data.Dataset: A TensorFlow Dataset containing the test data.

               Note:
                   This method returns the test dataset, which is suitable for use in
                   evaluating the performance of a trained machine learning model.

               Examples:
                   # Get the test dataset
                   test_dataset = Window_Gen.test()
               """
        self.BATCH_SIZE = self.BATCH_SIZE
        self.SHUFFLE = False
        test_window = self.make_dataset(self.test_df)
        test_window = test_window.map(self.split_windows_map)
        test_window = test_window.map(self.transform_element)

        return test_window.prefetch(tf.data.experimental.AUTOTUNE)
    
    def plot(self, inputs, labels, title, predictions = None, plot_col='Demand', max_subplots=3):
      
      plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      
      for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(np.arange(self.input_width )[self.input_slice] , 
                 inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)
    
        label_col_index = self.label_columns_indices.get(plot_col, None)
       
    
        plt.scatter(np.arange(self.total_window_size)[self.labels_slice],
                    labels[n, :,],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if predictions is not None:

          plt.scatter(np.arange(self.total_window_size)[self.labels_slice], 
                      predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)
    
        if n == 0:
          plt.legend()
          plt.title(f' Predictions for the take number: {title}')
      plt.xlabel('Time [h]')



BATCH_SIZE = 32
Window_Gen = SequencedData(
    input_width = 24*2*7,
    label_width = 24*1,
    SHIFT = 1,
    train_df = df_norm_train,
    test_df = df_norm_test,
    val_df = df_norm_val,
    label_columns = ['Demand'],
    BATCH_SIZE = BATCH_SIZE,
    input_start = 1,
    n_supplementary_features = len(['Demand']),
    SHUFFLE = True)

windowed_train = Window_Gen.train()
windowed_val = Window_Gen.val()
windowed_test = Window_Gen.test()

for i in windowed_train.take(1):
    (enc, dec_in), dec_out = i

print('All shapes are: (batch, time, features)')
print(f'Encoder inputs shape: {enc.shape}')
print(f'Decoder inputs shape: {dec_in.shape}')
print(f'Decoder outputs shape: {dec_out.shape}')
print("-*-*-*-*-*")
# print('All the sequences for take one')
# print(f'Encoder input: {enc[0][0]}')
# print(f'Decoder input: {dec_in[0]}')
# print(f'Decoder out: {dec_out[0]}')

input_shape = enc.shape[1:]  # Shape of input observations tensor
output_shape = dec_in.shape[1:]  # Shape of target labels tensor