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
	def __init__(self):
		self.pos = []
		self.quat = []

	def __init__(self, pos, quat):
		self.pos = []
		self.quat = []
		if len(pos) != 3:
			print("Error: Wrong Pos data len")
		for p in pos:
			if not isfloat(p):
				print("Error: Wrong Pos data type", pos)
			self.pos.append(float(p))
		if len(quat) != 4:
			print("Error: Wrong Quaternion data length")
		for q in quat:
			if not isfloat(q):
				print("Error: Wrong Quaternion data type", q)
			self.quat.append(float(q))

	def print_joint(self):
		print("	Joint: ", end='')
		for p in self.pos:
			print(p, end=' ')
		for q in self.quat:
			print(q, end=' ')
		print("")

class Skeleton:
	def __init__(self, data):
		self.lower_l = [0, 12, 13, 14, 15]
		self.lower_r = [0, 16, 17, 18, 19]
		self.lower_l_len = []
		self.lower_r_len = []

		self.joints = []
		for d in data:
			self.joints.append(Joint(d[0:3],d[3:]))
		self.cal_joint_to_joint_length()

	def print_skeleton(self):
		print("Skeleton: ")
		for joint in self.joints:
			joint.print_joint()

	def joint_length(self, A_idx, B_idx):
		return get_distance(self.joints[A_idx].pos, self.joints[B_idx].pos)

	def cal_joint_to_joint_length(self):
		for i in range(len(self.lower_l)-1):
			self.lower_l_len.append(self.joint_length(self.lower_l[i], self.lower_l[i+1]))
		for i in range(len(self.lower_r)-1):
			self.lower_r_len.append(self.joint_length(self.lower_r[i], self.lower_r[i+1]))

	def get_measurement(self):
		tmp = []
		for i in range(len(self.joints)):
			tmp = tmp + [self.joints[i].pos]
		return tmp

	def get_lower_measurement(self):
		tmp = []
		lower_point = [0, 12, 13, 14, 15, 16, 17, 18, 19]
		for i in lower_point:
			tmp.append(self.joints[i].pos)
		return tmp


