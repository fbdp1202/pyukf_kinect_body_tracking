import numpy as np
from skeleton import Skeleton
from scipy.spatial.transform import Rotation as R
import math

import time
from functools import wraps

import os

def check_time(function):
	@wraps(function)
	def measure(*args, **kwargs):
		start_time = time.time()
		result = function(*args, **kwargs)
		end_time = time.time()
		print(f"@check_time: {function.__name__} took {end_time - start_time}")
		return result

	return measure

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
		for i in range(x_min, x_max+1, 5):
			for j in range(y_min, y_max+1, 5):
				for k in range(z_min, z_max+1, 5):
					rot_mat = R.from_euler('xyz', [i,j,k], degrees=True).as_matrix()
					rot_mat = np.matmul(offset, rot_mat)
					BposW = np.matmul(rot_mat, AposB) + np.array(A_pos).reshape(3,1)
					cur_dist = self.get_dist(B_pos, BposW)
					if cur_dist < min_dist:
						min_dist = cur_dist
						min_degree = [i,j,k]
						n_offset = rot_mat
		return n_offset, min_degree

	@check_time
	def get_init_degree_lower_brute_force(self, data):
		print("get_init_degree_lower_brute_force:")
		root_p = np.array(data.joints[0].pos).reshape(-1)
		root_u = np.array([0,0,0])

		l_offset = np.eye(3)
		r_offset = np.eye(3)

		l_offset, root_l_t = self.get_opt_degree(data.joints[0].pos, data.joints[18].pos, 0, 0, 0, 360, 0, 360, l_offset)
		root_l_w = np.array([0,0,0])
		print("root_l_t = ", root_l_t)

		l_offset, hip_l_t = self.get_opt_degree(data.joints[18].pos, data.joints[19].pos, 0, 0, 0, 360, 0, 360, l_offset)
		hip_l_w = np.array([0,0,0])
		print("hip_l_t = ", hip_l_t)

		l_offset, knee_l_t = self.get_opt_degree(data.joints[19].pos, data.joints[20].pos, 0, 0, 0, 360, 0, 0, l_offset)
		knee_l_t = [knee_l_t[1]]
		knee_l_w = np.array([0])
		print("knee_l_t = ", knee_l_t)

		l_offset, ankle_l_t = self.get_opt_degree(data.joints[20].pos, data.joints[21].pos, 0, 0, 0, 360, 0, 360, l_offset)
		ankle_l_t = ankle_l_t[1:3]
		ankle_l_w = np.array([0,0])
		print("ankle_l_t = ", ankle_l_t)

		r_offset, root_r_t = self.get_opt_degree(data.joints[0].pos, data.joints[22].pos, 0, 0, 0, 360, 0, 360, r_offset)
		root_r_w = np.array([0,0,0])
		print("root_r_t = ", root_r_t)

		r_offset, hip_r_t = self.get_opt_degree(data.joints[22].pos, data.joints[23].pos, 0, 0, 0, 360, 0, 360, r_offset)
		hip_r_w = np.array([0,0,0])
		print("hip_r_t = ", hip_r_t)

		r_offset, knee_r_t = self.get_opt_degree(data.joints[23].pos, data.joints[24].pos, 0, 0, 0, 360, 0, 0, r_offset)
		knee_r_t = [knee_r_t[1]]
		knee_r_w = np.array([0])
		print("knee_r_t = ", knee_r_t)

		r_offset, ankle_r_t = self.get_opt_degree(data.joints[24].pos, data.joints[25].pos, 0, 0, 0, 360, 0, 360, r_offset)
		ankle_r_t = ankle_r_t[1:3]
		ankle_r_w = np.array([0,0])
		print("ankle_r_t = ", ankle_r_t)
		return np.concatenate((root_p,root_u,root_l_t,root_l_w,hip_l_t,hip_l_w,knee_l_t,knee_l_w,ankle_l_t,ankle_l_w,root_r_t,root_r_w,hip_r_t,hip_r_w,knee_r_t,knee_r_w,ankle_r_t,ankle_r_w), axis=0)

	@check_time
	def get_init_degree_upper_brute_force(self, data):
		print("get_init_degree_upper_brute_force:")
		root_p = data.joints[0].pos
		root_v = np.array([0,0,0])

		s_root_off, s_root_t = self.get_opt_degree(data.joints[0].pos, data.joints[1].pos, 0, 0, 0, 360, 0, 360, np.eye(3))
		s_root_w = np.array([0,0,0])
		print("s_root_t = ", s_root_t)
		
		spine_naval_off, spine_naval_t = self.get_opt_degree(data.joints[1].pos, data.joints[2].pos, 0, 0, 0, 360, 0, 360, s_root_off)
		spine_naval_w = np.array([0,0,0])
		print("spine_naval_t = ", spine_naval_t)
		
		l_spine_chest_off, l_spine_chest_t = self.get_opt_degree(data.joints[2].pos, data.joints[4].pos, 0, 0, 0, 360, 0, 360, spine_naval_off)
		l_spine_chest_w = np.array([0,0,0])
		print("l_spine_chest_t = ", l_spine_chest_t)
		
		l_shoulder_off, l_shoulder_t = self.get_opt_degree(data.joints[4].pos, data.joints[5].pos, 0, 0, 0, 360, 0, 360, l_spine_chest_off)
		l_shoulder_w = np.array([0,0,0])
		print("l_shoulder_t = ", l_shoulder_t)
		
		l_shoulder_center_off, l_shoulder_center_t = self.get_opt_degree(data.joints[5].pos, data.joints[6].pos, 0, 0, 0, 360, 0, 360, l_shoulder_off)
		l_shoulder_center_w = np.array([0,0,0])
		print("l_shoulder_center_t = ", l_shoulder_center_t)
		
		l_elbow_off, l_elbow_t = self.get_opt_degree(data.joints[6].pos, data.joints[7].pos, 0, 0, 0, 360, 0, 360, l_shoulder_center_off)
		l_elbow_w = np.array([0,0,0])
		print("l_elbow_t = ", l_elbow_t)
		
		l_wrist_u_off, l_wrist_u_t = self.get_opt_degree(data.joints[7].pos, data.joints[8].pos, 0, 0, 0, 360, 0, 360, l_elbow_off)
		l_wrist_u_w = np.array([0,0,0])
		print("l_wrist_u_t = ", l_wrist_u_t)
		
		l_hand_off, l_hand_t = self.get_opt_degree(data.joints[8].pos, data.joints[9].pos, 0, 0, 0, 360, 0, 360, l_wrist_u_off)
		l_hand_w = np.array([0,0,0])
		print("l_hand_t = ", l_hand_t)
		
		l_wrist_d_off, l_wrist_d_t = self.get_opt_degree(data.joints[7].pos, data.joints[10].pos, 0, 0, 0, 360, 0, 360, l_elbow_off)
		l_wrist_d_w = np.array([0,0,0])
		print("l_wrist_d_t = ", l_wrist_d_t)
		
		r_spine_chest_off, r_spine_chest_t = self.get_opt_degree(data.joints[2].pos, data.joints[11].pos, 0, 0, 0, 360, 0, 360, spine_naval_off)
		r_spine_chest_w = np.array([0,0,0])
		print("r_spine_chest_t = ", r_spine_chest_t)
		
		r_shoulder_off, r_shoulder_t = self.get_opt_degree(data.joints[11].pos, data.joints[12].pos, 0, 0, 0, 360, 0, 360, r_spine_chest_off)
		r_shoulder_w = np.array([0,0,0])
		print("r_shoulder_t = ", r_shoulder_t)
		
		r_shoulder_center_off, r_shoulder_center_t = self.get_opt_degree(data.joints[12].pos, data.joints[13].pos, 0, 0, 0, 360, 0, 360, r_shoulder_off)
		r_shoulder_center_w = np.array([0,0,0])
		print("r_shoulder_center_t = ", r_shoulder_center_t)
		
		r_elbow_off, r_elbow_t = self.get_opt_degree(data.joints[13].pos, data.joints[14].pos, 0, 0, 0, 360, 0, 360, r_shoulder_center_off)
		r_elbow_w = np.array([0,0,0])
		print("r_elbow_t = ", r_elbow_t)
		
		r_wrist_u_off, r_wrist_u_t = self.get_opt_degree(data.joints[14].pos, data.joints[15].pos, 0, 0, 0, 360, 0, 360, r_elbow_off)
		r_wrist_u_w = np.array([0,0,0])
		print("r_wrist_u_t = ", r_wrist_u_t)
		
		r_hand_off, r_hand_t = self.get_opt_degree(data.joints[15].pos, data.joints[16].pos, 0, 0, 0, 360, 0, 360, r_wrist_u_off)
		r_hand_w = np.array([0,0,0])
		print("r_hand_t = ", r_hand_t)
		
		r_wrist_d_off, r_wrist_d_t = self.get_opt_degree(data.joints[14].pos, data.joints[17].pos, 0, 0, 0, 360, 0, 360, r_elbow_off)
		r_wrist_d_w = np.array([0,0,0])
		print("r_wrist_d_t = ", r_wrist_d_t)
		
		u_spine_chest_off, u_spine_chest_t = self.get_opt_degree(data.joints[2].pos, data.joints[3].pos, 0, 0, 0, 360, 0, 360, spine_naval_off)
		u_spine_chest_w = np.array([0,0,0])
		print("u_spine_chest_t = ", u_spine_chest_t)
		
		neck_off, neck_t = self.get_opt_degree(data.joints[3].pos, data.joints[26].pos, 0, 0, 0, 360, 0, 360, u_spine_chest_off)
		neck_w = np.array([0,0,0])
		print("neck_t = ", neck_t)
		
		head_off, head_t = self.get_opt_degree(data.joints[26].pos, data.joints[27].pos, 0, 360, 0, 360, 0, 0, neck_off)
		head_w = np.array([0,0,0])
		print("head_t = ", head_t)
		
		l_nose_off, l_nose_t = self.get_opt_degree(data.joints[27].pos, data.joints[28].pos, 0, 0, 0, 360, 0, 360, head_off)
		l_nose_w = np.array([0,0,0])
		print("l_nose_t = ", l_nose_t)
		
		l_eye_off, l_eye_t = self.get_opt_degree(data.joints[28].pos, data.joints[29].pos, 0, 0, 0, 360, 0, 360, l_nose_off)
		l_eye_w = np.array([0,0,0])
		print("l_eye_t = ", l_eye_t)
		
		r_nose_off, r_nose_t = self.get_opt_degree(data.joints[27].pos, data.joints[30].pos, 0, 0, 0, 360, 0, 360, head_off)
		r_nose_w = np.array([0,0,0])
		print("r_nose_t = ", r_nose_t)
		
		r_eye_off, r_eye_t = self.get_opt_degree(data.joints[30].pos, data.joints[31].pos, 0, 0, 0, 360, 0, 360, r_nose_off)
		r_eye_w = np.array([0,0,0])
		print("r_eye_t = ", r_eye_t)
		return np.concatenate((root_p,root_v,s_root_t,s_root_w,spine_naval_t,spine_naval_w,l_spine_chest_t,l_spine_chest_w,l_shoulder_t,l_shoulder_w,l_shoulder_center_t,l_shoulder_center_w,l_elbow_t,l_elbow_w,l_wrist_u_t,l_wrist_u_w,l_hand_t,l_hand_w,l_wrist_d_t,l_wrist_d_w,r_spine_chest_t,r_spine_chest_w,r_shoulder_t,r_shoulder_w,r_shoulder_center_t,r_shoulder_center_w,r_elbow_t,r_elbow_w,r_wrist_u_t,r_wrist_u_w,r_hand_t,r_hand_w,r_wrist_d_t,r_wrist_d_w,u_spine_chest_t,u_spine_chest_w,neck_t,neck_w,head_t,head_w,l_nose_t,l_nose_w,l_eye_t,l_eye_w,r_nose_t,r_nose_w,r_eye_t,r_eye_w), axis=0)

	def get_init_degree_brute_force(self, data, filename):
		init_lower_mean = self.get_init_degree_lower_brute_force(data)
		init_upper_mean = self.get_init_degree_upper_brute_force(data)

		f = open(filename, 'w')
		for d in init_lower_mean:
			f.write(str(d) + ' ')
		f.write('\n')
		for d in init_upper_mean:
			f.write(str(d) + ' ')
		f.write('\n')
		f.close()

		return init_lower_mean, init_upper_mean

	@check_time
	def get_init_degree_cash_mode(self, data, filename):
		filename = filename.replace('.txt', '_init_mean.txt')

		if os.path.isfile(filename):
			f = open(filename, 'r')
			init_lower_mean = np.array(f.readline().replace(' \n', '').split(' '))
			for i in range(init_lower_mean.shape[0]):
				init_lower_mean[i] = float(init_lower_mean[i])

			init_upper_mean = np.array(f.readline().replace(' \n', '').split(' '))
			for i in range(init_upper_mean.shape[0]):
				init_upper_mean[i] = float(init_upper_mean[i])

			return init_lower_mean, init_upper_mean

		print(filename)
		return self.get_init_degree_brute_force(data, filename)

	def get_init_degree(self, idx, filename):
		return self.get_init_degree_cash_mode(self.data[idx], filename)

	@check_time
	def get_length_avg(self):
		cbr_num = len(self.data)
		mean_joint_to_joints = np.array([0]*31)
		for d in self.data:
			mean_joint_to_joints = mean_joint_to_joints + np.array(d.joint_to_joints)
		mean_joint_to_joints = mean_joint_to_joints / cbr_num
		mean_joint_to_joints = mean_joint_to_joints.tolist()
		mean_joint_to_joints_upper = mean_joint_to_joints[:23]
		mean_joint_to_joints_lower = mean_joint_to_joints[23:31]
		# lower_l_len_avg = np.array(lower_l_len).reshape(cbr_num,4).mean(axis=0)
		return mean_joint_to_joints_upper, mean_joint_to_joints_lower

	@check_time
	def get_init_mean(self, idx, filename):
		# lower_init_frontward = np.zeros(42)
		# upper_init_frontward = np.zeros(144)
		lower_init_frontward, upper_init_frontward = self.get_init_degree(idx, filename)
		joint_to_joints_upper, joint_to_joints_lower = self.get_length_avg()
		lower_init_mean = np.concatenate((lower_init_frontward, joint_to_joints_lower), axis=0)
		upper_init_mean = np.concatenate((upper_init_frontward, joint_to_joints_upper), axis=0)
		return lower_init_mean, upper_init_mean
