import math
import numpy as np
import pykalman
from MJU_params import MJU_Lower_Params
from scipy.spatial.transform import Rotation as R

class MJU_Filter:
	def __init__(self, param):
		self.transition_fun = self.transition_lower
		self.observation_fun = self.observation_lower
		self.trans_cov = param.trans_cov
		self.obs_cov = param.obs_cov
		self.mean = param.mean
		self.cov = param.init_trans_cov
		
		self.trnas_matrix = param.trans_matrx
		self.new_state = []

		self.lower_ukf = pykalman.AdditiveUnscentedKalmanFilter(
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

	def update_lower(self, measurement):
		self.mean, self.cov = self.lower_ukf.filter_update(self.mean, self.cov, measurement)
		return self.new_state

class MJU_Filter_Controler:
	def __init__(self, init_mean, init_cov):
		self.lower_point = [0, 12, 13, 14, 15, 16, 17, 18, 19]
		self.flt = MJU_Filter(MJU_Lower_Params(init_mean, init_cov))

	def get_lower_measurement(self, measurement):
		l_meas = []
		for i in self.lower_point:
			l_meas.append(measurement[i])
		return np.array(l_meas).reshape(-1)

	# update state using ukf
	def update(self, measurement):
		return self.flt.update_lower(self.get_lower_measurement(measurement))
