import abc
import numpy as np

from typing import Tuple


class BaseFeature(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns features (B, D) and coords (B, 3)
        """
        raise NotImplementedError
