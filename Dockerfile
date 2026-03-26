# Dockerfile
FROM python:3.12

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

RUN pip install mlflow

# Simulate downloading the model using the Run ID
RUN echo "Downloading model for Run ID: ${RUN_ID}"

CMD ["python", "-c", "import os; print(f'Serving model for Run ID: {os.environ[\"RUN_ID\"]}')"]