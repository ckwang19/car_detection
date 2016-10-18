from __future__ import print_function
import tensorflow as tf
import numpy as np


rows = np.array([0, 3], dtype=np.intp)
columns = np.array([0, 2], dtype=np.intp)

print rows[:, np.newaxis]
