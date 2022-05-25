from torch.utils.data.sampler import Sampler


class RepeatIterator(object):
    """
    creates an iterable which returns each integer in range(length) n_times times.
    example: next(RepeatIterator(2,3)) would return: 0,0,1,1,2,2,3,3
    # todo check what is rly returned!
    """
    def __init__(self, n_times, length):
        self.n_times = n_times
        self.length = length
        self.idx = 0
        self.reps = 0

    def __iter__(self):
        return self

    def __next__(self):
        #print(f"Number of TTA views Iterator: {self.n_times}")
        if self.reps < self.n_times:
            self.reps += 1
        else:
            self.idx += 1
            self.reps = 1

        if self.idx < self.length:
            return self.idx
        else:
            raise StopIteration


class RepeatedSampler(Sampler):
    """
    Sampler class that wraps the RepeatIterator. Can be used in dataloaders.
    """
    def __init__(self, n_times, data_source):
        self.n_times = n_times
        self.ds_length = len(data_source)
        super().__init__(data_source=data_source)
        #print(f"Number of TTA views Sampler: {self.n_times}")
        #print(f"ds length Sampler: {self.ds_length}")
        self.iterator = RepeatIterator(self.n_times, self.ds_length)

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return int(self.ds_length)*self.n_times

