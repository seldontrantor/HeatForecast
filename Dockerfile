# Dockerfile
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /VersionControl_Heat

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./utils ./utils
COPY ./datasets ./datasets
COPY load_and_norm.py .
COPY lstm_tensorflow_model.py .
COPY predict_heat.py .
COPY sequenced_data.py .

CMD ["python","sequenced_data.py"]