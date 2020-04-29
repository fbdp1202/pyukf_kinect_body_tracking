import math

def get_distance(A, B):
	total = 0
	for i in range(len(A)):
		total = total + (A[i]-B[i])*(A[i]-B[i])
	return math.sqrt(total)
