import numpy as np
from skeleton import Skeleton
from scipy.spatial.transform import Rotation as R
import math

class Calibration:
	def __init__(self):
		self.data = []

	def __init__(self, data):
		self.data = data

	def square(self, x):
		return x * x

	def get_dist(self, A_pos, B_pos):
		ret = 0
		for i in range(3):
			ret = ret + self.square(A_pos[i]-B_pos[i])
		return math.sqrt(ret)

	def get_opt_degree(self, A_pos, B_pos, x_min, x_max, y_min, y_max, z_min, z_max, offset):
		length = self.get_dist(A_pos, B_pos)
		AposB = np.array([length, 0, 0]).reshape(3,1)
		min_dist= 1e9
		min_degree = [0,0,0]
		BposW = []
		for i in range(x_min, x_max+1):
			for j in range(y_min, y_max+1):
				for k in range(z_min, z_max+1):
					rot_mat = R.from_euler('xyz', [i,j,k], degrees=True).as_matrix()
					rot_mat = np.matmul(offset, rot_mat)
					BposW = np.matmul(rot_mat, AposB) + np.array(A_pos).reshape(3,1)
					cur_dist = self.get_dist(B_pos, BposW)
					if cur_dist < min_dist:
						min_dist = cur_dist
						min_degree = [i,j,k]
						n_offset = rot_mat
		return n_offset, min_degree

	def get_lower_frontward_bruth_force(self, data):
		init_data = data
		lower_l = [0, 12, 13, 14, 15]
		lower_r = [0, 16, 17, 18, 19]
		lower_l_pos = []
		lower_r_pos = []
		for i in lower_l:
			lower_l_pos.append(init_data.joints[i].pos)
		for i in lower_r:
			lower_r_pos.append(init_data.joints[i].pos)

		root_p = np.array(init_data.joints[0].pos).reshape(-1)
		root_u = np.array([0,0,0])

		l_offset = np.eye(3)
		r_offset = np.eye(3)

		l_offset, root_l_t = self.get_opt_degree(lower_l_pos[0], lower_l_pos[1], 0, 0, 289, 289, 26, 26, l_offset)
		root_l_w = np.array([0,0,0])

		l_offset, hip_l_t = self.get_opt_degree(lower_l_pos[1], lower_l_pos[2], 0, 0, 46, 46, 87, 87, l_offset)
		hip_l_w = np.array([0,0,0])

		l_offset, knee_l_t = self.get_opt_degree(lower_l_pos[2], lower_l_pos[3], 0, 0, 308, 308, 0, 0, l_offset)
		knee_l_t = [knee_l_t[1]]
		knee_l_w = np.array([0])

		l_offset, ankle_l_t = self.get_opt_degree(lower_l_pos[3], lower_l_pos[4], 0, 0, 66, 66, 6, 6, l_offset)
		ankle_l_t = ankle_l_t[1:3]
		ankle_l_w = np.array([0,0])

		r_offset, root_r_t = self.get_opt_degree(lower_r_pos[0], lower_r_pos[1], 0, 0, 71, 71, 206, 206, r_offset)
		root_r_w = np.array([0,0,0])

		r_offset, hip_r_t = self.get_opt_degree(lower_r_pos[1], lower_r_pos[2], 0, 0, 46, 46, 271, 271, r_offset)
		hip_r_w = np.array([0,0,0])

		r_offset, knee_r_t = self.get_opt_degree(lower_r_pos[2], lower_r_pos[3], 0, 0, 316, 316, 0, 0, r_offset)
		knee_r_t = [knee_r_t[1]]
		knee_r_w = np.array([0])

		r_offset, ankle_r_t = self.get_opt_degree(lower_r_pos[3], lower_r_pos[4], 0, 0, 113, 113, 185, 185, r_offset)
		ankle_r_t = ankle_r_t[1:3]
		ankle_r_w = np.array([0,0])
		return np.concatenate((root_p,root_u,root_l_t,root_l_w,hip_l_t,hip_l_w,knee_l_t,knee_l_w,ankle_l_t,ankle_l_w,root_r_t,root_r_w,hip_r_t,hip_r_w,knee_r_t,knee_r_w,ankle_r_t,ankle_r_w), axis=0)


	def get_lower_frontward_quat(self, data):
		init_data = data
		lower_l = [0, 12, 13, 14, 15]
		lower_r = [0, 16, 17, 18, 19]
		lower_l_quat = []
		lower_r_quat = []
		for i in lower_l:
			lower_l_quat.append(init_data.joints[i].quat)
		for i in lower_r:
			lower_r_quat.append(init_data.joints[i].quat)
		lower_l_euler = R.from_quat(lower_l_quat).as_euler('xyz')
		lower_r_euler = R.from_quat(lower_r_quat).as_euler('xyz')

		root_p = np.array(init_data.joints[0].pos).reshape(-1)
		root_u = np.array([0,0,0])
		root_l_t = np.array(R.from_euler('xyz', lower_l_euler[1]).as_euler('xyz')[0:3]).reshape(-1)
		root_l_w = np.array([0,0,0])
		root_r_t = np.array(R.from_euler('xyz', lower_r_euler[1]).as_euler('xyz')[0:3]).reshape(-1)
		root_r_w = np.array([0,0,0])
		hip_l_t = np.array(R.from_euler('xyz', lower_l_euler[2] - lower_l_euler[1]).as_euler('xyz')[0:3]).reshape(-1)
		hip_l_w = np.array([0,0,0])
		knee_l_t = np.array(R.from_euler('xyz', lower_l_euler[3] - lower_l_euler[2]).as_euler('xyz')[2]).reshape(-1)
		knee_l_w = np.array([0])
		ankle_l_t = np.array(R.from_euler('xyz', lower_l_euler[4] - lower_l_euler[3]).as_euler('xyz')[1:3]).reshape(-1)
		ankle_l_w = np.array([0,0])
		hip_r_t = np.array(R.from_euler('xyz', lower_r_euler[2] - lower_r_euler[1]).as_euler('xyz')[0:3]).reshape(-1)
		hip_r_w = np.array([0,0,0])
		knee_r_t = np.array(R.from_euler('xyz', lower_r_euler[3] - lower_r_euler[2]).as_euler('xyz')[2]).reshape(-1)
		knee_r_w = np.array([0])
		ankle_r_t = np.array(R.from_euler('xyz', lower_r_euler[4] - lower_r_euler[3]).as_euler('xyz')[1:3]).reshape(-1)
		ankle_r_w = np.array([0,0])
		return np.concatenate((root_p,root_u,root_l_t,root_l_w,root_r_t,root_r_w,hip_l_t,hip_l_w,knee_l_t,knee_l_w,ankle_l_t,ankle_l_w,hip_r_t,hip_r_w,knee_r_t,knee_r_w,ankle_r_t,ankle_r_w), axis=0)

	def get_lower_frontward(self, idx, mode):
		if mode == 'bruth_force':
			return self.get_lower_frontward_bruth_force(self.data[idx])
		else:
			return self.get_lower_frontward_quat(self.data[idx])

	def get_lower_length_avg(self):
		cbr_num = len(self.data)
		lower_l_len = []
		lower_r_len = []
		for i in range(cbr_num):
			lower_l_len = lower_l_len + self.data[i].lower_l_len
			lower_r_len = lower_r_len + self.data[i].lower_r_len
		lower_l_len_avg = np.array(lower_l_len).reshape(cbr_num,4).mean(axis=0)
		lower_r_len_avg = np.array(lower_r_len).reshape(cbr_num,4).mean(axis=0)
		return np.concatenate((lower_l_len_avg, lower_r_len_avg), axis=0)		

	def get_init_mean(self):
		lower_init_frontward = self.get_lower_frontward(50, 'bruth_force')
		lower_init_backward = self.get_lower_length_avg()
		return np.concatenate((lower_init_frontward, lower_init_backward), axis=0)
