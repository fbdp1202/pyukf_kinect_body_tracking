import math
import numpy as np
import pykalman
from IJU_params import IJU_Lower_Params
from scipy.spatial.transform import Rotation as R

class IJU_Filter:
	def __init__(self, prams, here):
		self.here = here

		self.transition_fun = self.trans_fun
		if here == 0:
			self.observation_fun = self.obs_pos_fun
		else:
			self.observation_fun = self.obs_degree_fun

		self.trans_cov = prams.trans_cov[here]
		self.obs_cov = prams.obs_cov[here]
		self.mean = prams.mean[here]
		self.cov = prams.init_trans_cov[here]

		self.ukf = pykalman.AdditiveUnscentedKalmanFilter(
			self.transition_fun,
			self.observation_fun, 
			self.trans_cov, 
			self.obs_cov, 
			self.mean, 
			self.cov
		)

		self.state_dim = (int)(len(self.mean)/2)
		self.trans_mat = prams.trans_matrx[here]

		self.prev_pos = []
		self.cur_pos = []
		self.offset_rot_mat = np.eye(3)

		self.next_filter = []
		if here in prams.graph:
			for nx in prams.graph[here]:
				self.next_filter.append(Filter(prams, nx))

	def trans_fun(self, state):
		return np.matmul(self.trans_mat, np.array(state).reshape(len(state),1)).reshape(-1)

	def obs_pos_fun(self, state):
		self.cur_pos = state[0:3]
		for flt in self.next_filter:
			flt.prev_pos = state[0:3]
		return state[0:3]

	def obs_degree_fun(self, state):
		cur_link_length = state[self.state_dim*2]
		cur_vec = np.array([cur_link_length,0,0]).reshape(3,1)
		cur_degree = state[:self.state_dim]
		if self.state_dim == 3:
			rot_mat = R.from_euler('xyz', cur_degree, degrees=True).as_matrix()
		elif self.state_dim == 2:
			rot_mat = R.from_euler('yz', cur_degree, degrees=True).as_matrix()
		else:
			rot_mat = R.from_euler('y', cur_degree, degrees=True).as_matrix()

		rot_mat = np.matmul(self.offset_rot_mat, rot_mat)
		self.cur_pos = np.array(np.matmul(rot_mat, cur_vec) + np.array(self.prev_pos).reshape(3,1)).reshape(-1)
		for flt in self.next_filter:
			flt.prev_pos = self.cur_pos
			flt.offset_rot_mat = rot_mat

		return self.cur_pos

	def update(self, measurement):
		self.mean, self.cov = self.ukf.filter_update(self.mean, self.cov, measurement[self.here])
		ret = self.cur_pos
		for flt in self.next_filter:
			ret = np.concatenate((ret, flt.update(measurement)), axis=0)
		return ret

class IJU_Filter_Controler:
	def __init__(self, init_mean, init_cov):
		self.lower_point = [0, 12, 13, 14, 15, 16, 17, 18, 19]
		self.new_state = {}
		self.flt = IJU_Filter(IJU_Lower_Params(init_mean, init_cov), 0)

	def get_lower_measurement(self, measurement):
		l_meas = {}
		for i in self.lower_point:
			l_meas[i] = measurement[i]
		return l_meas

	# update state using ukf
	def update(self, measurement):
		return self.flt.update(self.get_lower_measurement(measurement))
