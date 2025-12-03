import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("PATH_CUSUM"))

import numpy as np
from config.setting import GeneratorSetting, CUSUMSetting, ProbCUSUMSetting
from source.generator.change_point_generator import ChangePointGenerator
from source.detector.cusum import CUSUM_Detector, ProbCUSUM_Detector, ChartCUSUM_Detector
from loguru import logger

def main():

    logger.info("CUSUM algorithms")
    logger.info("Generating data with change points")
    dpg_params = GeneratorSetting()
    # Generate time series data with change points
    generator = ChangePointGenerator(num_segments=dpg_params.num_segments,
                                    segment_length=dpg_params.segment_length,
                                    change_point_type=dpg_params.change_point_type,
                                    seed=dpg_params.seed
                                    )
    generator.generate_data()
    # Plot the generated data
    generator.plot_data()

    # Detect change points using CUSUM Detector
    cusum_params = CUSUMSetting()
    cusum_detector = CUSUM_Detector(warmup_period= cusum_params.warmup_period,
                                    delta=  cusum_params.delta, 
                                    threshold=cusum_params.threshold)
    cusum_pos_changes, cusum_neg_changes, cusum_change_points = cusum_detector.detect_change_points(np.array(generator.data))
    # Plot the detected change points using CUSUM Detector
    cusum_detector.plot_change_points(generator.data, cusum_change_points, cusum_pos_changes, cusum_neg_changes)


    # Detect change points using Probabilistic CUSUM Detector
    prob_cusum_setting = ProbCUSUMSetting()
    prob_cusum_detector = ProbCUSUM_Detector(warmup_period=prob_cusum_setting.warmup_period, 
                                             threshold_probability=prob_cusum_setting.threshold_probability)
    prob_probabilities, prob_change_points = prob_cusum_detector.detect_change_points(np.array(generator.data))
    # Plot the detected change points using Probabilistic CUSUM Detector
    prob_cusum_detector.plot_change_points(generator.data, prob_change_points, prob_probabilities)




if __name__ == "__main__":
    main()
