import sys
sys.path.append('./code/')
from utils import *

def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

class Joint:
	def __init__(self, pos):
		self.pos = []
		if len(pos) != 3:
			print("Error: Wrong Pos data len")
		for p in pos:
			if not isfloat(p):
				print("Error: Wrong Pos data type", pos)
			self.pos.append(float(p))

class Skeleton:
	def __init__(self, data):
		self.joints = []
		self.graph = self.skeleton_graph()
		self.sk_num = 32
		for i in range(self.sk_num):
			self.joints.append(Joint(data[i][0:3]))

		# upper_length = self.joint_to_joints[:25]
		# lower_length = self.joint_to_joints[25:34]
		self.joint_to_joints = self.cal_joint_to_joint_length()

	def skeleton_graph(self):
		graph = {}
		graph[0] = [1, 18, 22]
		graph[1] = [2]
		graph[2] = [4, 11, 3]
		# left hand
		graph[4] = [5]
		graph[5] = [6]
		graph[6] = [7]
		graph[7] = [8,10]
		graph[8] = [9]
		# right hand
		graph[11] = [12]
		graph[12] = [13]
		graph[13] = [14]
		graph[14] = [15, 17]
		graph[15] = [16]
		# left foot
		graph[18] = [19]
		graph[19] = [20]
		graph[20] = [21]
		# right foot
		graph[22] = [23]
		graph[23] = [24]
		graph[24] = [25]
		# head
		graph[3] = [26]
		graph[26] = [27]
		graph[27] = [28, 30]
		graph[28] = [29]
		graph[30] = [31]
		return graph		

	def joint_length(self, x, y):
		return get_distance(self.joints[x].pos, self.joints[y].pos)

	def travel_joint_to_joint(self, here):
		tmp = []
		for nx in self.graph[here]:
			tmp.append(self.joint_length(here, nx))
			if nx in self.graph:
				tmp = tmp + self.travel_joint_to_joint(nx)
		return tmp

	def cal_joint_to_joint_length(self):
		return self.travel_joint_to_joint(0)

	def get_measurement(self):
		tmp = []
		for i in range(len(self.joints)):
			tmp = tmp + [self.joints[i].pos]
		return tmp


