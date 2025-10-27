from abc import ABC, abstractmethod
from bisect import bisect_right
from collections import defaultdict
from typing import Optional, Sequence, Union

import numpy as np


class AlphaScheduler(ABC):
    curr_step: int = 0

    def step(self):
        self.curr_step += 1

    def reset(self):
        self.curr_step = 0

    @abstractmethod
    def get_alpha(self):
        pass


class SteppingAlpha(AlphaScheduler):
    """Given lists [a(0), a(1), ..., a(n)], [s(1), ..., s(n)], sets the alpha at the current step to a(i) iff s(i-1) <= step < s(i)

    Args:
        alphas (Union[Sequence[int], Sequence[float]]): Sequence of alphas, where the first alpha is the initial value
        steps (Sequence[int] | Sequence[float]): Sequence of steps at which the respective alpha should be set.
            If the steps are passed as floats, they will be interpreted as
            1. Fractions of the total training length if units='steps',
            2. Fractional epochs if units='epochs'.
        units (str, optional): Either 'steps' or 'epochs'
        max_epochs (int, optional): Maximum number of epochs to train
        steps_per_epoch (int, optional): Number of steps per epoch. Required if units are set to `epochs` or
            passed as string.
    """

    def __init__(
        self,
        alphas: Union[Sequence[int], Sequence[float]],
        steps: Sequence[int],
        units: Optional[str] = "steps",
        max_epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        assert len(alphas) == len(steps) + 1
        steps = np.array(steps)
        is_ascending = len(steps) == 1 or all([a < b for a, b in zip(steps, steps[1:])])
        if not is_ascending:
            raise ValueError("Values in `steps` are not in ascending order!")

        def _count_vals(iterable):
            counts = defaultdict(lambda: 0)
            for val in iterable:
                counts[val] += 1
            return counts

        has_reps = any([cnt > 1 for cnt in _count_vals(steps).values()])
        if has_reps:
            raise ValueError("Found repeting values in `steps`!")
        if np.issubdtype(steps.dtype, np.float):
            # Convert floats to ints
            if units == "steps":
                # Interpret as fractions of full training duration
                steps = np.ceil(max_epochs * steps_per_epoch * steps).astype(int)
            elif units == "epochs":
                # Interpret as fractions of an epoch
                steps = np.ceil(steps_per_epoch * steps).astype(int)
        elif units == "epochs":
            steps = steps_per_epoch * steps

        self.alphas = alphas
        self.steps = steps

    def get_alpha(self):
        pos = bisect_right(self.steps, self.curr_step)
        return self.alphas[pos]


class ExponentialAlpha(AlphaScheduler):
    """Decays an initial alpha by gamma every epoch until last_epoch is reached.
        When last_epoch=-1, never stops decaying alpha.

    Args:
        alpha (float): Initial alpha.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    """

    def __init__(self, alpha: float, gamma: float, last_step=-1, **kwargs):
        super().__init__()
        self.alpha0 = alpha
        self.gamma = gamma
        self.last_step = last_step

    def get_alpha(self):
        step = (
            self.curr_step
            if (self.last_step == -1 or self.curr_step < self.last_step)
            else self.last_step
        )
        return self.alpha0 * self.gamma ** float(step)
