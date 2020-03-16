import numpy as np

from .base import MultivariateOutlierGenerator


class MultivariateExtremeOutlierGenerator(MultivariateOutlierGenerator):
    def __init__(self, timestamps=None, factor_range=(8, 10)):
        self.timestamps = [] if timestamps is None else list(sum(timestamps, ()))
        if factor_range[0] >= factor_range[1]:
            raise ValueError('First value must be less than second value')
        self.min_factor = factor_range[0]
        self.factor_width = factor_range[1] - factor_range[0]

    def get_value(self, current_timestamp, timeseries):
        if current_timestamp in self.timestamps:
            local_std = timeseries.iloc[max(0, current_timestamp - 10):current_timestamp + 10].std()
            random_factor = np.random.random() * self.factor_width + self.min_factor
            # import pdb; pdb.set_trace()
            return np.random.choice([-1, 1]) * random_factor * local_std
        else:
            return 0

    def add_outliers(self, timeseries):
        # print(self.timestamps)
        additional_values = []
        for timestamp_index in range(len(timeseries)):
            # print(timestamp_index)
            # print(timestamp_index in self.timestamps)
            additional_values.append(self.get_value(timestamp_index, timeseries))
        return additional_values
