#!/bin/bash

airflow db migrate

airflow users create \
  --username "$AIRFLOW_USERNAME" \
  --password "$AIRFLOW_PASSWORD" \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com || true

exec airflow standalone