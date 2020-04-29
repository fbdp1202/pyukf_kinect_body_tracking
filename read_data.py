import re
import numpy as np

def read_data_skeleton(filename):
	f = open(filename, 'r')
	lines = f.readlines()
	l = len(lines)
	data = []
	for line in lines:
		refix = re.findall('-*[0-9]+.[0-9]+e*-*[0-9]*', line)
		data = data + refix

	if filename == 'input_stand.txt':
		data = np.array(data).reshape((int)(l/26),26,7)
	else:
		data = np.array(data).reshape(l,26,7)

	return data
