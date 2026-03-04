import pandas as pd
from detector.cusum import CUSUM_Detector
from loguru import logger
import pathlib

dir_path = pathlib.Path(__file__).parent.parent

try:
    data_path = dir_path / "data" / "synthetic_data.csv"
    data = pd.read_csv(data_path)
    logger.info("Data read successfully.")
except Exception as e:
    logger.error(f"Error reading data: {e}")
    raise e