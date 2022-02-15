# MLCF modules
from mlcf.datatools.wtseries import WTSeries


class WTSeriesPreProcess():
    """Preprocessing class for WTSeries objects
    """
    def __init__(self, data: WTSeries):
        self.data = data
        if not isinstance(self.data, WTSeries):
            raise TypeError("data must be a WTSeries")


class Identity(WTSeriesPreProcess):
    def __call__(self, *args, **kwargs):
        return self.data


class AutoNormalize(WTSeriesPreProcess):
    def __call__(self, *args, **kwargs):
        for i in range(len(self.data)):
            self.data[i] = ((self.data[i] - self.data[i].mean()) / self.data[i].std()).round(6)
        return self.data


PreProcessDict = {
    "Identity": Identity,
    "AutoNormalize": AutoNormalize,
    None: Identity
}
