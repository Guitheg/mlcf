from typing import TypeVar
# MLCF modules
from mlcf.datatools.wtseries import WTSeries
from multiprocessing import Pool, cpu_count


N_CPU = cpu_count()


class WTSeriesPreProcess:
    """Preprocessing class for WTSeries objects"""

    def __init__(self, data: WTSeries):
        self.data = data.copy()
        if not isinstance(self.data, WTSeries):
            raise TypeError("data must be a WTSeries")


class Identity(WTSeriesPreProcess):
    def __call__(self, *args, **kwargs) -> WTSeries:
        return self.data


def normalize(window):
    normalized_window = (window - window.mean()) / window.std()
    if not normalized_window.isnull().values.any():
        return normalized_window
    return None


class AutoNormalize(WTSeriesPreProcess):
    def __call__(self, *args, **kwargs) -> WTSeries:
        with Pool(processes=N_CPU) as pl:
            list_normalized_windows = [
                elem for elem in pl.map(normalize, self.data) if elem is not None
            ]
        self.data.data = list_normalized_windows
        return self.data


WTSeriesPreProcessType = TypeVar("WTSeriesPreProcessType", bound=WTSeriesPreProcess)

PreProcessDict = {"Identity": Identity, "AutoNormalize": AutoNormalize, None: Identity}
