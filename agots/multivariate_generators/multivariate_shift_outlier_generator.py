import numpy as np

from .base import MultivariateOutlierGenerator


class MultivariateShiftOutlierGenerator(MultivariateOutlierGenerator):
    def __init__(self, timestamps=None, factor_range=(8, 10)):
        if factor_range[0] >= factor_range[1]:
            raise ValueError('First value must be less than second value')
        timestamps = timestamps or []
        self.timestamps = timestamps
        self.min_factor = factor_range[0]
        self.factor_width = factor_range[1] - factor_range[0]

    def add_outliers(self, timeseries):
        additional_values = np.zeros(timeseries.size)
        for start, end in self.timestamps:
            local_std = timeseries.iloc[max(0, start - 10):end + 10].std()
            random_factor = np.random.random() * self.factor_width + self.min_factor
            additional_values[list(range(start, end))] += np.random.choice([-1, 1]) * random_factor * local_std
        return additional_values
