import math
import numpy as np
import pykalman
from MJU_params import MJU_Lower_Params
from scipy.spatial.transform import Rotation as R

class MJU_Filter:
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
		self.new_state = []

		self.ukf = pykalman.AdditiveUnscentedKalmanFilter(
			self.transition_fun,
			self.observation_fun,
			self.trans_cov,
			self.obs_cov,
			self.mean,
			self.cov
		)

	# transition function for Additive UKF
	# constant position
	def transition_lower(self, state):
		return np.matmul(self.trnas_matrix, state)

	# observation function for Additive UKF
	# expected range data
	def observation_lower(self, state):
		wPr = np.array(state[0:3]).reshape(3,1)									#[0:6]
		wRr_l = R.from_euler('xyz',state[6:9], degrees=True).as_matrix()		#[6:12]
		rRh_l = R.from_euler('xyz', state[12:15], degrees=True).as_matrix()		#[12:18]
		hRk_l = R.from_euler('y', state[18], degrees=True).as_matrix()			#[18:20]
		kRa_l = R.from_euler('yz', state[20:22], degrees=True).as_matrix()		#[20:24]
		wRr_r = R.from_euler('xyz',state[24:27], degrees=True).as_matrix()		#[24:30]
		rRh_r = R.from_euler('xyz', state[30:33], degrees=True).as_matrix()		#[30:36]
		hRk_r = R.from_euler('y', state[36], degrees=True).as_matrix()			#[36:38]
		kRa_r = R.from_euler('yz', state[38:40], degrees=True).as_matrix()		#[38:42]

		wRh_l = np.matmul(wRr_l, rRh_l)
		wRk_l = np.matmul(wRh_l, hRk_l)
		wRa_l = np.matmul(wRk_l, kRa_l)

		wRh_r = np.matmul(wRr_r, rRh_r)
		wRk_r = np.matmul(wRh_r, hRk_r)
		wRa_r = np.matmul(wRk_r, kRa_r)

		rPh_l = np.array([state[42],0,0]).reshape(3,1)
		hPk_l = np.array([state[43],0,0]).reshape(3,1)
		kPa_l = np.array([state[44],0,0]).reshape(3,1)
		aPf_l = np.array([state[45],0,0]).reshape(3,1)

		rPh_r = np.array([state[46],0,0]).reshape(3,1)
		hPk_r = np.array([state[47],0,0]).reshape(3,1)
		kPa_r = np.array([state[48],0,0]).reshape(3,1)
		aPf_r = np.array([state[49],0,0]).reshape(3,1)

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
		root = np.array(state[0:3]).reshape(3,1)										#[0:6]
		s_root = R.from_euler('xyz', state[6:9], degrees=True).as_matrix()				#[6:12]
		spine_naval = R.from_euler('xyz', state[12:15], degrees=True).as_matrix()		#[12:18]
		l_spine_chest = R.from_euler('xyz', state[18:21], degrees=True).as_matrix()		#[18:24]
		l_shoulder = R.from_euler('xyz', state[24:27], degrees=True).as_matrix()		#[24:30]
		l_shoulder_center = R.from_euler('xyz', state[30:33], degrees=True).as_matrix()	#[30:36]
		l_elbow = R.from_euler('y', state[36], degrees=True).as_matrix()				#[36:38]
		r_spine_chest = R.from_euler('xyz', state[38:41], degrees=True).as_matrix()		#[38:44]
		r_shoulder = R.from_euler('xyz', state[44:47], degrees=True).as_matrix()		#[44:50]
		r_shoulder_center = R.from_euler('xyz', state[50:53], degrees=True).as_matrix()	#[50:56]
		r_elbow = R.from_euler('y', state[58], degrees=True).as_matrix()				#[56:58]
		u_spine_chest = R.from_euler('xyz', state[58:61], degrees=True).as_matrix()		#[58:64]
		neck = R.from_euler('xyz', state[64:67], degrees=True).as_matrix()				#[64:70]

		s_root = s_root
		spine_naval = np.matmul(s_root, spine_naval)
		l_spine_chest = np.matmul(spine_naval, l_spine_chest)
		l_shoulder = np.matmul(l_spine_chest, l_shoulder)
		l_shoulder_center = np.matmul(l_shoulder, l_shoulder_center)
		l_elbow = np.matmul(l_shoulder_center, l_elbow)
		r_spine_chest = np.matmul(spine_naval, r_spine_chest)
		r_shoulder = np.matmul(r_spine_chest, r_shoulder)
		r_shoulder_center = np.matmul(r_shoulder, r_shoulder_center)
		r_elbow = np.matmul(r_shoulder_center, r_elbow)
		u_spine_chest = np.matmul(spine_naval, u_spine_chest)
		neck = np.matmul(u_spine_chest, neck)

		L_root_spine = np.array([state[70],0,0]).reshape(3,1)
		L_spine_chest = np.array([state[70],0,0]).reshape(3,1)

		L_l_chest_sh = np.array([state[70],0,0]).reshape(3,1)
		L_l_sh_shc = np.array([state[70],0,0]).reshape(3,1)
		L_l_shc_elbow = np.array([state[70],0,0]).reshape(3,1)
		L_l_elbow_wrist = np.array([state[70],0,0]).reshape(3,1)

		L_r_chest_sh = np.array([state[70],0,0]).reshape(3,1)
		L_r_sh_shc = np.array([state[70],0,0]).reshape(3,1)
		L_r_shc_elbow = np.array([state[70],0,0]).reshape(3,1)
		L_r_elbow_wrist = np.array([state[70],0,0]).reshape(3,1)

		L_chest_neck = np.array([state[70],0,0]).reshape(3,1)
		L_neck_head = np.array([state[70],0,0]).reshape(3,1)

		root_p = root
		spine_naval_p = np.matmul(s_root, L_root_spine) + root_p
		spine_chest_p = np.matmul(spine_naval, L_spine_chest) + spine_naval_p

		l_shoulder_p = np.matmul(l_spine_chest, L_l_chest_sh) + spine_chest_p
		l_shoulder_center_p = np.matmul(l_shoulder, L_l_sh_shc) + l_shoulder_p
		l_elbow_p = np.matmul(l_shoulder_center, L_l_shc_elbow) + l_shoulder_center_p
		l_wrist_p = np.matmul(l_elbow, L_l_elbow_wrist) + l_elbow_p

		r_shoulder_p = np.matmul(r_spine_chest, L_r_chest_sh) + spine_chest_p
		r_shoulder_center_p = np.matmul(r_shoulder, L_r_sh_shc) + r_shoulder_p
		r_elbow_p = np.matmul(r_shoulder_center, L_r_shc_elbow) + r_shoulder_center_p
		r_wrist_p = np.matmul(r_elbow, L_r_elbow_wrist) + r_elbow_p

		neck_p = np.matmul(u_spine_chest, L_chest_neck) + spine_chest_p
		head_p = np.matmul(neck, L_neck_head) + neck_p

		self.new_state = np.concatenate((root_p,spine_naval_p,spine_chest_p,l_shoulder_p,l_shoulder_center_p,l_elbow_p,l_wrist_p,r_shoulder_p,r_shoulder_center_p,r_elbow_p,r_wrist_p,neck_p,head_p), axis=0).reshape(-1)
		return self.new_state

	def update(self, measurement):
		self.mean, self.cov = self.ukf.filter_update(self.mean, self.cov, measurement)
		return self.new_state

class MJU_Filter_Controler:
	def __init__(self, l_init_mean, l_init_cov, u_init_mean, u_init_cov):
		self.lower_point = [0, 12, 13, 14, 15, 16, 17, 18, 19]
		self.upper_point = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 3, 20]
		self.l_flt = MJU_Filter(MJU_Lower_Params(l_init_mean, l_init_cov), 'lower')
		self.u_flt = MJU_Filter(MJU_Upper_Params(u_init_mean, u_init_cov), 'upper')

	def get_measurement(self, measurement, list_point):
		l_meas = []
		for i in list_point:
			l_meas.append(measurement[i])
		return np.array(l_meas).reshape(-1)
	def get_new_state(self, lower_state, upper_state):


	# update state using ukf
	def update(self, measurement):
		lower_state = self.l_flt.update_lower(self.get_measurement(measurement, self.lower_point))
		upper_state = self.u_flt.upper_
