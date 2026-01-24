from abc import ABC, abstractmethod

class Cusum(ABC):

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _update_data(self):
        pass

    @abstractmethod
    def _init_params(self):
        pass

    @abstractmethod
    def _detect_changepoint(self):
        pass

    @abstractmethod
    def detection(self):
        pass

    @abstractmethod
    def offline_detection(self):
        pass

    @abstractmethod
    def plot_change_points(self):
        pass

