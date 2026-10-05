# Use Airflow 3.3.1 with Python 3.11.
FROM apache/airflow:3.3.1-python3.11

# Copy project source code.
COPY src /opt/wine_quality_prediction/src
COPY config /opt/wine_quality_prediction/config
COPY artifacts /opt/wine_quality_prediction/artifacts

# Make the project source importable.
ENV PYTHONPATH=/opt/wine_quality_prediction/src
