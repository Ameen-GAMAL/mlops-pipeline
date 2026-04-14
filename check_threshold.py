# check_threshold.py
import mlflow
import sys

THRESHOLD = 0.99


def check():
    # Read the Run ID from the artifact
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    print(f"Checking Run ID: {run_id}")

    # Connect to MLflow and fetch the run
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)

    print(f"Accuracy from MLflow: {accuracy}")

    if accuracy < THRESHOLD:
        print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
        sys.exit(1)
    else:
        print(f"PASSED: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}")


if __name__ == "__main__":
    check()