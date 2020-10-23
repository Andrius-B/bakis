from src.runners.run_parameters import RunParameters
from collections import OrderedDict

class SearcherRunParamsProvider:
    def __init__(self, param_ranges: OrderedDict, default_parameter_factory):
        self.param_factory = default_parameter_factory
        self.param_ranges = param_ranges
        self.counters = [0 for _ in self.param_ranges]
        self.complete = False
    
    def __iter__(self):
        self.counters = [0 for _ in self.param_ranges]
        self.complete = False
        return self

    def __next__(self):
        if self.complete:
            raise StopIteration
        params = self.__create_run_params()
        last = self.__increment_counter(0)
        if last:
            self.complete = True
        print(f"Returning params: {params.all_params}")
        return params
        
    def __len__(self):
        acc = 1
        for x in self.param_ranges.values():
            acc = acc * len(x)
        return acc

    def __create_run_params(self):
        params: RunParameters = self.param_factory()
        search_overrides = {}
        for i, key in enumerate(self.param_ranges.keys()):
            param_value = self.param_ranges[key][self.counters[i]]
            search_overrides[key] = param_value

        params.apply_overrides(search_overrides)
        return params

    def __increment_counter(self, index):
        if len(self.counters) == index:
            return True
        param_ranges_at_index = self.param_ranges[list(self.param_ranges.keys())[index]]

        if self.counters[index] >= (len(param_ranges_at_index) - 1):
            # overflow in the counter
            self.counters[index] = 0
            return self.__increment_counter(index + 1)
        else:
            self.counters[index] = self.counters[index] + 1
            return False


class SearcherRunParamsProviderBuilder:

    def __init__(self, default_parameter_factory):
        self.param_factory = default_parameter_factory
        self.param_ranges = OrderedDict()

    def add_param(self, name, param_range):
        self.param_ranges[name] = param_range
        return self
    
    def build(self) -> SearcherRunParamsProvider:
        return SearcherRunParamsProvider(self.param_ranges, self.param_factory)
