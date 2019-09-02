from typing import Tuple, Any, List
from torch import Tensor
from PIL import Image

image_size = Tuple[int, int]
generator_input = Tuple[Tensor, Tensor]
gan_input = Tuple[Tensor, bool]
epoch_output = Tuple[float, Any]
sample_output = List[Any]


def no_except(func, *args, **kwargs):
    try:
        # print(func, str(*args), str(**kwargs))
        func(*args, **kwargs)
    except Exception:
        pass
