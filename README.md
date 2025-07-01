# ForHeat

ForHeat (**For**ecasting **Heat** Demand) contains utilities and example models for short term heat demand forecasting using TensorFlow/Keras.  The project demonstrates how to prepare time series data, train LSTM encoder–decoder networks and evaluate the resulting models.

The repository is not packaged for direct installation but all code can be executed from the cloned sources.

## Repository layout

```
.
├── datasets/              Sample demand and weather data
├── load_and_norm.py       Helper functions for loading and normalising datasets
├── lstm_tensorflow_model.py  Training script for the LSTM model
├── predict_heat.py        Evaluate a trained model and produce metrics
├── sequenced_data.py      Utilities for windowing time series into sequences
├── utils/                 Additional helper modules
└── requirements.txt       Python dependencies
```

## Getting started

1. **Clone the repository**

   ```bash
   git clone https://github.com/AminDar/HeatForecast.git
   cd HeatForecast
   ```

2. **Install dependencies** (preferably inside a virtual environment)

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**

   The repository ships with example data under `datasets/`.  The `load_and_normalize` function in `load_and_norm.py` will
   split the data set, optionally remove future‑unknown features and apply Min–Max scaling.

   ```python
   from load_and_norm import load_and_normalize

   dataset = "datasets/df_sin_cosing.csv"
   train_df, test_df, val_df, df_train_norm, df_test_norm, df_val_norm, scaler = \
       load_and_normalize(dataset, columns_to_normalize=["Demand", "Temp"])
   ```

4. **Create windowed datasets**

   `SequencedData` in `sequenced_data.py` generates TensorFlow datasets suitable for an encoder–decoder model.  A convenience
   function `load_default_data()` builds these datasets using the provided CSV files.

   ```python
   from sequenced_data import load_default_data
   data = load_default_data()
   windowed_train = data["windowed_train"]
   windowed_val = data["windowed_val"]
   windowed_test = data["windowed_test"]
   ```

## Training a model

`lstm_tensorflow_model.py` trains an LSTM encoder–decoder network.  The script takes optional arguments for the number of epochs and the learning rate:

```bash
python lstm_tensorflow_model.py --nepoch 200 --learning_rate 0.001
```

The model along with the fitted scaler and training history are stored in a timestamped subdirectory of `weights/`.

## Making predictions

Use `predict_heat.py` to load a saved model and compute forecast metrics.  By default the script picks the latest model in `weights/`:

```bash
python predict_heat.py --model-path weights/<timestamp>/whole_model.keras
```

Metrics for each 24 hour period and the overall results are written to CSV files inside `Metrics/` and a line plot summarising the metrics is saved alongside them.

## Example output

When training completes an example forecast plot is produced similar to the one below.

![Forecast Example](weights/07-Feb-2024-16-36-32/1-1.png)

## Dataset description

`datasets/df_sin_cosing.csv` contains hourly demand together with several engineered time and weather features.  Two raw source files `dhn_demand.csv` and `weather_denmark.csv` are also provided.  `utils/preprocessing_eda.py` demonstrates how these can be merged and enriched with cyclic time features, holidays and scaling utilities for exploratory analysis.

## Contributing

Contributions are welcome via pull requests.  Please open an issue first to discuss substantial changes.

## License

This project is licensed under the terms of the MIT License.  See the [LICENSE](LICENSE) file for details.

