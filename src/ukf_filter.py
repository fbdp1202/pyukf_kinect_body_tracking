import math
import numpy as np
import pykalman
from ukf_params import *
from scipy.spatial.transform import Rotation as R

class ukf_Filter:
	def __init__(self, param, mode):
		self.transition_fun = self.transition_lower
		if mode == 'lower':
			self.observation_fun = self.observation_lower
		else:
			self.observation_fun = self.observation_upper

		self.trans_cov = param.trans_cov
		self.obs_cov = param.obs_cov
		self.mean = param.mean
		self.cov = param.init_trans_cov
		
		self.trnas_matrix = param.trans_matrx
		self.cur_state = []
		self.new_state = []

		self.ukf = pykalman.AdditiveUnscentedKalmanFilter(
			self.transition_fun,
			self.observation_fun,
			self.trans_cov,
			self.obs_cov,
			self.mean,
			self.cov
		)

	def euler_to_rot_matrix(self, state, pos):
		return R.from_euler(pos, state, degrees=True).as_matrix()

	# transition function for Additive UKF
	# constant position
	def transition_lower(self, state):
		return np.matmul(self.trnas_matrix, state)

	# observation function for Additive UKF
	# expected range data
	def observation_lower(self, state):
		self.cur_state = state
#######################################################################################################################
# read state
#######################################################################################################################
		wPr   = np.array(state[0:3]).reshape(3,1)					#[0:6]

		wRr_l = self.euler_to_rot_matrix(state[6:9], 'xyz')			#[6:12]
		rRh_l = self.euler_to_rot_matrix(state[12:15], 'xyz')		#[12:18]
		hRk_l = self.euler_to_rot_matrix(state[18], 'y')			#[18:20]
		kRa_l = self.euler_to_rot_matrix(state[20:22], 'yz')		#[20:24]
		wRr_r = self.euler_to_rot_matrix(state[24:27], 'xyz')		#[24:30]
		rRh_r = self.euler_to_rot_matrix(state[30:33], 'xyz')		#[30:36]
		hRk_r = self.euler_to_rot_matrix(state[36], 'y')			#[36:38]
		kRa_r = self.euler_to_rot_matrix(state[38:40], 'yz')		#[38:42]

#######################################################################################################################
# Get each joint rotation matrix
#######################################################################################################################
		wRh_l = np.matmul(wRr_l, rRh_l)
		wRk_l = np.matmul(wRh_l, hRk_l)
		wRa_l = np.matmul(wRk_l, kRa_l)

		wRh_r = np.matmul(wRr_r, rRh_r)
		wRk_r = np.matmul(wRh_r, hRk_r)
		wRa_r = np.matmul(wRk_r, kRa_r)

#######################################################################################################################
# Define length vector
#######################################################################################################################
		rPh_l = np.array([state[42],0,0]).reshape(3,1)
		hPk_l = np.array([state[43],0,0]).reshape(3,1)
		kPa_l = np.array([state[44],0,0]).reshape(3,1)
		aPf_l = np.array([state[45],0,0]).reshape(3,1)

		rPh_r = np.array([state[46],0,0]).reshape(3,1)
		hPk_r = np.array([state[47],0,0]).reshape(3,1)
		kPa_r = np.array([state[48],0,0]).reshape(3,1)
		aPf_r = np.array([state[49],0,0]).reshape(3,1)

#######################################################################################################################
# get joint position using transformation
# Equation: (child joint position) = (rotation matrix) * (parent to child length vector) + (parent joint position)
# Equation: cur_pos = rot_mat * [sk_len, 0, 0] + prev_pos
#######################################################################################################################
		wPr   = wPr
		wPh_l = np.matmul(wRr_l, rPh_l) + wPr
		wPk_l = np.matmul(wRh_l, hPk_l) + wPh_l
		wPa_l = np.matmul(wRk_l, kPa_l) + wPk_l
		wPf_l = np.matmul(wRa_l, aPf_l) + wPa_l

		wPh_r = np.matmul(wRr_r, rPh_r) + wPr
		wPk_r = np.matmul(wRh_r, hPk_r) + wPh_r
		wPa_r = np.matmul(wRk_r, kPa_r) + wPk_r
		wPf_r = np.matmul(wRa_r, aPf_r) + wPa_r

		self.new_state = np.concatenate((wPr, wPh_l, wPk_l, wPa_l, wPf_l, wPh_r, wPk_r, wPa_r, wPf_r), axis=0).reshape(-1)
		return self.new_state

	# observation function for Additive UKF
	# expected range data
	def observation_upper(self, state):
		self.cur_state = state
#######################################################################################################################
# read state
#######################################################################################################################
		root 				= np.array(state[0:3]).reshape(3,1)							#[0:6]

		s_root				= self.euler_to_rot_matrix(state[6:9], 	 'xyz')				#[6:12]
		spine_naval			= self.euler_to_rot_matrix(state[12:15], 'xyz')				#[12:18]

		l_spine_chest 		= self.euler_to_rot_matrix(state[18:21], 'xyz')				#[18:24]
		l_shoulder 			= self.euler_to_rot_matrix(state[24:27], 'xyz')				#[24:30]
		l_shoulder_center 	= self.euler_to_rot_matrix(state[30:33], 'xyz')				#[30:36]
		l_elbow 			= self.euler_to_rot_matrix(state[36:39], 'xyz')				#[36:42]
		l_wrist_u 			= self.euler_to_rot_matrix(state[42:45], 'xyz')				#[42:48]
		l_hand				= self.euler_to_rot_matrix(state[48:51], 'xyz')				#[48:54]
		l_wrist_d 			= self.euler_to_rot_matrix(state[54:57], 'xyz')				#[54:60]

		r_spine_chest 		= self.euler_to_rot_matrix(state[60:63], 'xyz')				#[60:66]
		r_shoulder 			= self.euler_to_rot_matrix(state[66:69], 'xyz')				#[66:72]
		r_shoulder_center 	= self.euler_to_rot_matrix(state[72:75], 'xyz')				#[72:78]
		r_elbow 			= self.euler_to_rot_matrix(state[78:81], 'xyz')				#[78:84]
		r_wrist_u 			= self.euler_to_rot_matrix(state[84:87], 'xyz')				#[84:90]
		r_hand				= self.euler_to_rot_matrix(state[90:93], 'xyz')				#[90:96]
		r_wrist_d 			= self.euler_to_rot_matrix(state[96:99], 'xyz')				#[96:102]

		u_spine_chest 		= self.euler_to_rot_matrix(state[102:105], 'xyz')			#[102:108]
		neck 				= self.euler_to_rot_matrix(state[108:111], 'xyz') 			#[108:114]
		head 				= self.euler_to_rot_matrix(state[114:117], 'xyz') 			#[114:120]
		l_nose 				= self.euler_to_rot_matrix(state[120:123], 'xyz') 			#[120:126]
		l_eye 				= self.euler_to_rot_matrix(state[126:129], 'xyz') 			#[126:132]
		r_nose 				= self.euler_to_rot_matrix(state[132:135], 'xyz') 			#[132:138]
		r_eye 				= self.euler_to_rot_matrix(state[138:141], 'xyz') 			#[138:144]

#######################################################################################################################
# Get each joint rotation matrix
#######################################################################################################################
		s_root 				= s_root
		spine_naval 		= np.matmul(s_root, 			spine_naval 			)

		l_spine_chest 		= np.matmul(spine_naval, 		l_spine_chest 			)
		l_shoulder 			= np.matmul(l_spine_chest, 		l_shoulder 				)
		l_shoulder_center 	= np.matmul(l_shoulder, 		l_shoulder_center 		)
		l_elbow 			= np.matmul(l_shoulder_center, 	l_elbow 				)
		l_wrist_u			= np.matmul(l_elbow, 			l_wrist_u 				)
		l_hand 				= np.matmul(l_wrist_u, 			l_hand 					)
		l_wrist_d 			= np.matmul(l_elbow, 			l_wrist_d 				)

		r_spine_chest 		= np.matmul(spine_naval, 		r_spine_chest 			)
		r_shoulder 			= np.matmul(r_spine_chest, 		r_shoulder 				)
		r_shoulder_center 	= np.matmul(r_shoulder, 		r_shoulder_center 		)
		r_elbow 			= np.matmul(r_shoulder_center, 	r_elbow 				)
		r_wrist_u			= np.matmul(r_elbow, 			r_wrist_u 				)
		r_hand 				= np.matmul(r_wrist_u, 			r_hand 					)
		r_wrist_d 			= np.matmul(r_elbow, 			r_wrist_d 				)

		u_spine_chest 		= np.matmul(spine_naval,  		u_spine_chest 			)
		neck 				= np.matmul(u_spine_chest,  	neck 					)

#######################################################################################################################
# Define length vector
#######################################################################################################################
		D_root_spine 		= np.array([state[144],0,0]).reshape(3,1)
		D_spine_chest 		= np.array([state[145],0,0]).reshape(3,1)

		D_l_chest_sh 		= np.array([state[146],0,0]).reshape(3,1)
		D_l_sh_shc 			= np.array([state[147],0,0]).reshape(3,1)
		D_l_shc_elbow 		= np.array([state[148],0,0]).reshape(3,1)
		D_l_elbow_wrist 	= np.array([state[149],0,0]).reshape(3,1)
		D_l_wrist_hand 		= np.array([state[150],0,0]).reshape(3,1)
		D_l_hand_handtip	= np.array([state[151],0,0]).reshape(3,1)
		D_l_wrist_thumb		= np.array([state[152],0,0]).reshape(3,1)

		D_r_chest_sh 		= np.array([state[153],0,0]).reshape(3,1)
		D_r_sh_shc 			= np.array([state[154],0,0]).reshape(3,1)
		D_r_shc_elbow 		= np.array([state[155],0,0]).reshape(3,1)
		D_r_elbow_wrist 	= np.array([state[156],0,0]).reshape(3,1)
		D_r_wrist_hand 		= np.array([state[157],0,0]).reshape(3,1)
		D_r_hand_handtip	= np.array([state[158],0,0]).reshape(3,1)
		D_r_wrist_thumb		= np.array([state[159],0,0]).reshape(3,1)

		D_chest_neck 		= np.array([state[160],0,0]).reshape(3,1)
		D_neck_head 		= np.array([state[161],0,0]).reshape(3,1)
		D_head_nose			= np.array([state[162],0,0]).reshape(3,1)
		D_l_nose_eye 		= np.array([state[163],0,0]).reshape(3,1)
		D_l_eye_ear			= np.array([state[164],0,0]).reshape(3,1)
		D_r_nose_eye		= np.array([state[165],0,0]).reshape(3,1)
		D_r_eye_ear			= np.array([state[166],0,0]).reshape(3,1)

#######################################################################################################################
# get joint position using transformation
# Equation: (child joint position) = (rotation matrix) * (parent to child length vector) + (parent joint position)
# Equation: cur_pos = rot_mat * [sk_len, 0, 0] + prev_pos
#######################################################################################################################
		root_p 				= root
		spine_naval_p 		= np.matmul(s_root, 			D_root_spine 	) + root_p
		spine_chest_p 		= np.matmul(spine_naval, 		D_spine_chest 	) + spine_naval_p

		l_shoulder_p 		= np.matmul(l_spine_chest, 		D_l_chest_sh	) + spine_chest_p
		l_shoulder_center_p = np.matmul(l_shoulder, 		D_l_sh_shc		) + l_shoulder_p
		l_elbow_p 			= np.matmul(l_shoulder_center, 	D_l_shc_elbow	) + l_shoulder_center_p
		l_wrist_p 			= np.matmul(l_elbow, 			D_l_elbow_wrist	) + l_elbow_p
		l_hand_p 			= np.matmul(l_wrist_u, 			D_l_wrist_hand	) + l_wrist_p
		l_handtip_p 		= np.matmul(l_hand, 			D_l_hand_handtip) + l_hand_p
		l_thumb_p 			= np.matmul(l_wrist_d, 			D_l_wrist_thumb	) + l_wrist_p

		r_shoulder_p 		= np.matmul(r_spine_chest, 		D_r_chest_sh	) + spine_chest_p
		r_shoulder_center_p = np.matmul(r_shoulder, 		D_r_sh_shc		) + r_shoulder_p
		r_elbow_p 			= np.matmul(r_shoulder_center, 	D_r_shc_elbow	) + r_shoulder_center_p
		r_wrist_p 			= np.matmul(r_elbow, 			D_r_elbow_wrist	) + r_elbow_p
		r_hand_p 			= np.matmul(r_wrist_u, 			D_r_wrist_hand	) + r_wrist_p
		r_handtip_p 		= np.matmul(r_hand, 			D_r_hand_handtip) + r_hand_p
		r_thumb_p 			= np.matmul(r_wrist_d, 			D_r_wrist_thumb	) + r_wrist_p

		neck_p 				= np.matmul(u_spine_chest, 		D_chest_neck	) + spine_chest_p
		head_p 				= np.matmul(neck, 				D_neck_head		) + neck_p
		nose_p 				= np.matmul(head, 				D_head_nose		) + head_p
		l_eye_p 			= np.matmul(l_nose, 			D_l_nose_eye	) + nose_p
		l_ear_p 			= np.matmul(l_eye, 				D_l_eye_ear		) + l_eye_p
		r_eye_p 			= np.matmul(r_nose, 			D_r_nose_eye	) + nose_p
		r_ear_p 			= np.matmul(r_eye, 				D_r_eye_ear		) + r_eye_p

		self.new_state = np.concatenate((root_p,spine_naval_p,spine_chest_p,l_shoulder_p,l_shoulder_center_p,l_elbow_p,l_wrist_p,l_hand_p,l_handtip_p,l_thumb_p,r_shoulder_p,r_shoulder_center_p,r_elbow_p,r_wrist_p,r_hand_p,r_handtip_p,r_thumb_p,neck_p,head_p,nose_p,l_eye_p,l_ear_p,r_eye_p,r_ear_p), axis=0).reshape(-1)
		return self.new_state

	def update(self, measurement):
		self.mean, self.cov = self.ukf.filter_update(self.mean, self.cov, measurement)
		return self.cur_state, self.new_state

class ukf_Filter_Controler:
	def __init__(self, l_init_mean, l_init_cov, u_init_mean, u_init_cov):
		self.lower_point = [0,18,19,20,21,22,23,24,25]
		self.upper_point = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,3,26,27,28,29,30,31]
		self.l_flt = ukf_Filter(ukf_Lower_Params(l_init_mean, l_init_cov), 'lower')
		self.u_flt = ukf_Filter(ukf_Upper_Params(u_init_mean, u_init_cov), 'upper')

	def get_measurement(self, measurement, list_point):
		l_meas = []
		for i in list_point:
			l_meas.append(measurement[i])
		return np.array(l_meas).reshape(-1)

	def get_cur_state(self, lower_state, upper_state):
		return list(lower_state) + list(upper_state)

	def get_new_state(self, lower_state, upper_state):
		new_dic = {}
		for i in range(len(self.lower_point)):
			idx = self.lower_point[i]
			new_dic[idx] = lower_state[i*3:i*3+3]

		for i in range(len(self.upper_point)):
			idx = self.upper_point[i]
			new_dic[idx] = upper_state[i*3:i*3+3]

		new_state = []
		for i in range(32):
			new_state.append(new_dic[i])

		return new_state

	# update state using ukf
	def update(self, measurement):
		lower_cur_state, lower_new_state = self.l_flt.update(self.get_measurement(measurement, self.lower_point))
		upper_cur_state, upper_new_state = self.u_flt.update(self.get_measurement(measurement, self.upper_point))
		return self.get_cur_state(lower_cur_state, upper_cur_state), self.get_new_state(lower_new_state, upper_new_state)
