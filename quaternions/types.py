"""Custom types and custom type hints.

Custom types
------------

.. data:: SCALAR
    scalar type

.. data:: VECTOR
    vector type

.. data:: POINT
    4D point type

Custom type hints
-----------------

.. data:: T_SCALAR
    type hint for scalar type

.. data:: T_VECTOR
    type hint for vector type

.. data:: T_POINT
    type hint for 4D point

.. data:: T_PAIR
    type hint for pair (scalar; 3D vector)

"""

from typing import List, Tuple, Union

import numpy as np

# custom type definitions
SCALAR = (int, float, np.number)
VECTOR = (List, np.ndarray)
POINT = (List, np.ndarray)

# custom type hints
T_SCALAR = float
T_VECTOR = Union[List[float], np.ndarray]
T_POINT = T_VECTOR
T_PAIR = Tuple[T_SCALAR, T_VECTOR]
