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

	data = np.array(data).reshape(l,32,7)
	return data

if __name__ == '__main__':
	data = read_data_skeleton('../data/standing_30sec.txt')
	print(data.shape)
