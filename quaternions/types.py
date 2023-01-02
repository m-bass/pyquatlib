from typing import List, Tuple, Union

import numpy as np

# types
SCALAR = (int, float, np.number)
VECTOR = (List, np.ndarray)
POINT = (List, np.ndarray)

# type hints
# T_SCALAR = Union[int, float, np.number]
T_SCALAR = float
# T_VECTOR = Union[List[T_SCALAR], np.ndarray]
T_VECTOR = Union[List[float], np.ndarray]
ALL_VECTOR = Union[float, T_VECTOR]
# ALL_VECTOR = Union[T_VECTOR, List[T_VECTOR], np.ndarray]
T_POINT = T_VECTOR
# T_POINT = Union[List[T_SCALAR], np.ndarray]
ALL_POINT = ALL_VECTOR
# ALL_POINT = Union[T_POINT, List[T_POINT], np.ndarray]
T_PAIR = Tuple[T_SCALAR, T_VECTOR]
ALL_PAIR = Union[T_PAIR, List[T_PAIR]]
