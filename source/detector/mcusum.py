from source.detector.cusum import CUSUM_Detector
from loguru import logger
import matplotlib.pyplot as plt

class OneShotCusum:
    def __init__(self, detector_name='cusum', **kwargs):
        self.detector = CUSUM_Detector(**kwargs) if detector_name == 'cusum' else None
        self.list_change_points = []

    def offline_detection(self, data_streams: list):
        dict_results = {}
        for i, stream in enumerate(data_streams):
            logger.info(f'Starting offline detection for stream {i+1}')
            result = self.detector.offline_detection(stream)
            dict_results[f'stream_{i+1}'] = result
            if 'change_points' in result:
                logger.info(f'Change points detected at: {result["change_points"]}')
                self.list_change_points.append(result['change_points'])
            logger.info(f'Finished offline detection\n')
        return dict_results
    
    def get_change_points(self):
        return self.list_change_points
    
    def plots_detection_many_streams(self, list_data_streams: list):
        axs = plt.subplots(len(list_data_streams), 1, figsize=(len(list_data_streams)*10, 5))[1]
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            ax.plot(list_data_streams[i], color='blue', label='Detected Change Points')
            if len(self.list_change_points[i]) != 0:
                for change_point in self.list_change_points[i]:
                    ax.vlines(change_point, 
                            ymin=min(list_data_streams[i]), 
                            ymax=max(list_data_streams[i]), 
                            colors='red', 
                            linestyles='dashed', 
                            label='Change Point' if change_point == self.list_change_points[i][0] else "")
            ax.set_title(f'Data Stream {i+1} - Detected Change Points')
            ax.set_xlabel('Time')
            ax.set_ylabel('Change Point Indicator')
            ax.legend()
        plt.tight_layout()
        plt.show()
