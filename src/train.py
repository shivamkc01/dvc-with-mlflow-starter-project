import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import warnings
import logging

# Ensure logs directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging to a file in the logs directory
log_file = os.path.join(log_dir, "training2.log")
logging.basicConfig(
    level=logging.INFO,  # Change to INFO
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler(sys.stdout)  # Log to console (optional)
    ]
)

logger = logging.getLogger(__name__)



# Get url from DVC
import dvc.api

path = "data/winequality-red.csv"
repo = "D:/shivam/ZenML-code"
version = "v2"

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
    )

mlflow.set_experiment("Demo")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(40)

    logger.info("Starting the training process...")
    # Read the wine-quality csv file from the remote repo
    logger.info("Reading data from DVC...")
    data = pd.read_csv(data_url, sep=',')
    logger.info(f"Data loaded successfully. Shape: {data.shape}")

    # Log data params
    mlflow.log_param('data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', data.shape[0])
    mlflow.log_param('input_cols', data.shape[1])

    # Split the data into training and testing sets (0.75, 0.25) split
    train, test = train_test_split(data)

    # The predicted column is 'quality' which is a scalar from (3,9)
    train_x = train.drop(['quality'], axis=1)
    test_x = test.drop(['quality'], axis=1)
    train_y = train[['quality']]
    test_y = test[['quality']]

    # Log artifacts columns used for modeling
    cols_x = pd.DataFrame(list(train_x.columns))
    cols_x.to_csv('outputs/features.csv', header=False, index=False)
    mlflow.log_artifact('outputs/features.csv')

    cols_y = pd.DataFrame(list(train_y.columns))
    cols_y.to_csv('outputs/targets.csv', header=False, index=False)
    mlflow.log_artifact('outputs/targets.csv')

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    logger.info("Training the ElasticNet model...")
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    logger.info("Model training completed.")

    # Make predictions on the test set
    predicted_qualities = lr.predict(test_x)

    # Evaluate the model
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)
    logger.info(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
    logger.info(f"  RMSE: {rmse}")
    logger.info(f"  MAE: {mae}")
    logger.info(f"  R2: {r2}")

    # Log parameters, metrics, and model to MLflow
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Create input example and infer signature
    input_example = train_x.head(1)  # Use a single row as input example
    signature = infer_signature(train_x, predicted_qualities)

    # Log the model with the signature and input example
    mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=input_example)

    logger.info(f"Model saved in run {mlflow.active_run().info.run_uuid}")
