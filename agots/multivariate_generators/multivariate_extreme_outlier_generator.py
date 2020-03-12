import numpy as np

from .base import MultivariateOutlierGenerator


class MultivariateExtremeOutlierGenerator(MultivariateOutlierGenerator):
    def __init__(self, timestamps=None, factor_range=(8, 10)):
        if factor_range[0] >= factor_range[1]:
            raise ValueError('First value must be less than second value')
        timestamps = timestamps or []
        self.timestamps = timestamps
        self.min_factor = factor_range[0]
        self.factor_width = factor_range[1] - factor_range[0]

    def get_value(self, current_timestamp, timeseries):
        if current_timestamp in self.timestamps:
            local_std = timeseries.iloc[max(0, current_timestamp - 10):current_timestamp + 10].std()
            random_factor = np.random.random() * self.factor_width + self.min_factor
            return np.random.choice([-1, 1]) * random_factor * local_std
        else:
            return 0

    def add_outliers(self, timeseries):
        additional_values = []
        for timestamp_index in range(len(timeseries)):
            additional_values.append(self.get_value(timestamp_index, timeseries))
        return additional_values
