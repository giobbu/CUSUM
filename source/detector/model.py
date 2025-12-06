from source.detector.cusum import CUSUM_Detector

class CusumModel:

    def __init__(self, detector_name='cusum', **kwargs):
        self.detector = CUSUM_Detector(**kwargs) if detector_name == 'cusum' else None

    def offline_detection(self, data_streams: list):
        results = []
        for stream in data_streams:
            result = self.detector.offline_detection(stream)
            results.append(result)
        return results