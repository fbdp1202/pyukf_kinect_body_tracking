import math
import numpy as np

def mean_squared_error(y, t, test_num):
	y = np.array(y).reshape(test_num, (int)(len(y)/test_num))
	t = np.array(t).reshape(test_num, (int)(len(t)/test_num))
	return ((y-t)**2).mean(axis=None)