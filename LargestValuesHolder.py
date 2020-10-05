class LargestValuesHolder:
    """
    A data structure to hold n elements with the largest metric values.
    """

    def __init__(self, n_elements=10):
        self._metric_values = {}
        self._elements = {}
        self._min_idx = 0

        # init metrics with 0s and elements with None
        for i in range(n_elements):
            self._metric_values[i] = 0
            self._elements[i] = None

    def get_min_value(self, with_index=False):
        """Return the minimum metric value. if with_index=True return its index to"""
        if with_index:
            return self._metric_values[self._min_idx], self._min_idx
        else:
            return self._metric_values[self._min_idx]

    def add_item(self, item, metric_value):
        """Add an item to the data structure. Put it in the place of the item with the minimum metric value"""
        min_value, min_idx = self.get_min_value(with_index=True)

        # check validity of the input
        if min_value > metric_value:
            return None
        else:
            self._metric_values[min_idx] = metric_value
            self._elements[min_idx] = item

            # search for new min value
            min_idx = self._min_idx
            for key in self._metric_values.keys():
                if self._metric_values[key] < self._metric_values[min_idx]:
                    min_idx = key
            self._min_idx = min_idx

    def get_items(self):
        """Return n highest scoring elements"""
        return list(self._elements.values())

    def get_max_value(self):
        """Return the maximum metric value"""
        max_idx = 0
        for key in self._metric_values.keys():
            if self._metric_values[key] > self._metric_values[max_idx]:
                max_idx = key
        return self._metric_values[max_idx]


if __name__ == '__main__':
    pass

