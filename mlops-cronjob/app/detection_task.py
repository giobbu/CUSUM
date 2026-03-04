import pathlib
import pandas as pd
import numpy as np
import pickle
from detector.cusum import CUSUM_Detector
from loguru import logger

dir_path = pathlib.Path(__file__).parent.parent
data_path = dir_path / "data" / "synthetic_data.csv"
results_path = dir_path / "data" / "detection_results.pkl"

def read_data(path):
    try:
        data = pd.read_csv(path)
        logger.info("Data read successfully.")
        logger.info(f"Data head: \n {data.head()}")
        return data
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        raise

def save_results(results, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results saved successfully to {path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

try:
    data = read_data(data_path)
    detector = CUSUM_Detector()
    results = detector.offline_detection(np.array(data["value"]))
    logger.info("CUSUM Detector fitted successfully.")
    save_results(results, results_path)
except Exception as e:
    logger.error(f"An error occurred: {e}")

