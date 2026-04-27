from abc import ABC, abstractmethod

class Cusum(ABC):
    """Abstract Base Class for CUSUM-based change point detectors."""

    @abstractmethod
    def _reset(self):
        """Reset the detector to its initial state."""
        pass

    @abstractmethod
    def _update_data(self):
        """Update the data used by the detector."""
        pass

    @abstractmethod
    def _init_params(self):
        """Initialize parameters for the detector."""
        pass

    @abstractmethod
    def _detect_changepoint(self):
        """Detect change points in the data."""
        pass

    @abstractmethod
    def detection(self):
        """Perform online change point detection."""
        pass

    @abstractmethod
    def offline_detection(self):
        """Perform offline change point detection."""
        pass

    @abstractmethod
    def plot_change_points(self):
        """Plot the detected change points."""
        pass

