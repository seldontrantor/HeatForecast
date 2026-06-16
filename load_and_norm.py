import os

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set_theme(style="dark")
sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})

def norm_data(
    train_df,
    test_df,
    val_df,
    scaler,
    normalize_all_features=True,
    columns_to_normalize=None,
):
    """
    Normalize specified columns in the input dataframes using the given scaler.

    :param train_df: pd.DataFrame, training dataset
    :param test_df: pd.DataFrame, test dataset
    :param val_df: pd.DataFrame, validation dataset
    :param scaler: A scikit-learn scaler instance (e.g., MinMaxScaler)
    :param normalize_all_features: bool, if True normalize all columns, otherwise use selected ones
    :param columns_to_normalize: list of column names to normalize when normalize_all_features is False
    :return: tuple of normalized train, test, val DataFrames and the fitted scaler
    """
    df_norm_train = train_df.copy()
    df_norm_test = test_df.copy()
    df_norm_val = val_df.copy()

    if normalize_all_features:
        columns_to_normalize = df_norm_train.columns

    scaler.fit(df_norm_train[columns_to_normalize])
    df_norm_train[columns_to_normalize] = scaler.transform(
        df_norm_train[columns_to_normalize]
    )
    df_norm_test[columns_to_normalize] = scaler.transform(
        df_norm_test[columns_to_normalize]
    )
    df_norm_val[columns_to_normalize] = scaler.transform(
        df_norm_val[columns_to_normalize]
    )

    return df_norm_train, df_norm_test, df_norm_val, scaler


def load_and_normalize(
    dataset_path: str,
    future_importance: bool = True,
    normalize_all_features: bool = False,
    columns_to_normalize=None,
    train_size: int = 8760,
    val_size: int = 8760,
    scaler=None,
):
    """Load a dataset, split it and normalize selected columns.

    Returns
    -------
    tuple
        (train_dfs, test_dfs, val_dfs, df_norm_train, df_norm_test, df_norm_val, scaler)
    """

    assert os.path.exists(dataset_path), f"File not found: {dataset_path}"

    df = pd.read_csv(dataset_path, parse_dates=[0, 15], index_col=0)
    df.drop(["date", "weekday"], axis=1, inplace=True)
    df = df.astype("float64")

    if future_importance:
        df.drop(
            [
                "quarter",
                "sin_dow",
                "cos_dow",
                "sin_dom",
                "cos_dom",
                "sin_year",
                "cos_year",
                "solar",
                "Holidays",
                "wind",
                "humid",
                "sin_doy",
                "cos_doy",
                "sin_month",
                "cos_month",
                "sin_woy",
                "cos_woy",
            ],
            axis=1,
            inplace=True,
        )

    train_dfs = df.iloc[:train_size, :]
    val_dfs = df.iloc[train_size : train_size + val_size, :]
    test_dfs = df.iloc[train_size + val_size :, :]

    if scaler is None:
        scaler = MinMaxScaler()

    df_norm_train, df_norm_test, df_norm_val, scaler = norm_data(
        train_dfs,
        test_dfs,
        val_dfs,
        scaler,
        normalize_all_features=normalize_all_features,
        columns_to_normalize=columns_to_normalize,
    )

    return (
        train_dfs,
        test_dfs,
        val_dfs,
        df_norm_train,
        df_norm_test,
        df_norm_val,
        scaler,
    )


if __name__ == "__main__":
    dataset_path = os.path.join("datasets", "df_sin_cosing.csv")
    train_dfs, test_dfs, val_dfs, df_norm_train, df_norm_test, df_norm_val, scaler = (
        load_and_normalize(dataset_path, columns_to_normalize=["Demand", "Temp"])
    )
    print("Train set shape:", train_dfs.shape)
    print("Validation set shape:", val_dfs.shape)
    print("Test set shape:", test_dfs.shape)
