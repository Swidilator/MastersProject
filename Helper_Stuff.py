from typing import Tuple, Any, List
from PIL import Image

image_size = Tuple[int, int]
epoch_output = Tuple[float, Any]
sample_output = List[Any]


def no_except(func, *args, **kwargs):
    try:
        print(func, str(*args), str(**kwargs))
        func(*args, **kwargs)
    except Exception:
        pass