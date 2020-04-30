import numpy as np

'''
Lower body model
	state vector
	x = [
		root: 			3D(6)	p(x,y,z), u(x,y,z), 	 	,
		l_root: 		3D(6)						theta(x,y,z), 	w(x,y,z),
		l-hip: 			3D(6) 						theta(x,y,z), 	w(x,y,z)	,
		l-knee: 		1D(2) 						theta(z), 		w(z)		,
		l-ankle: 		2D(4) 						theta(y,z), 	w(y,z)		,
        r_root:         3D(6)                       theta(x,y,z),   w(x,y,z),
		r-hip: 			3D(6)						theta(x,y,z), 	w(x,y,z)	,
		r-knee: 		1D(2)						theta(z), 		w(z)		,
		r-ankle: 		2D(4) 						theta(y,z), 	w(y,z)		,
		Ll-root-hip: 	1D(1)	L												,
		Ll-hip-knee: 	1D(1)	L												,
		Ll-knee-ankle: 	1D(1)	L												,
		Ll-ankle-foot: 	1D(1)	L												,
		Lr-root-hip: 	1D(1)	L												,
		Lr-hip-knee: 	1D(1)	L												,
		Lr-knee-ankle: 	1D(1)	L												,
		Lr-ankle-foot: 	1D(1)	L												,
	]

	total vector length(DP): 50

	measurement vector
	y = [
		x_root: 		3D(3)
		x_l_hip: 		3D(3)
		x_l-knee:		3D(3)
		x_l-ankle:		3D(3)
		x_l-foot:		3D(3)
		x_r-hip:		3D(3)
		x_r-knee:		3D(3)
		x_r-ankle:		3D(3)
		x_r-foot:		3D(3)
	]
	total vector length(MP): 27
'''
class MJU_Lower_Params:
    def __init__(self, init_mean, init_cov=[1e-6, 1e-4, 1e-6, 1e-4, 1e-6, 1e-1, 1e-6, 1e-1, 1e-6, 1e-4, 1e-4, 100]):
    	self.set_trans_covariance(init_cov[:10])
    	self.set_obs_covariance(init_cov[10])
    	self.set_mean(init_mean)
    	self.set_init_trans_cov(init_cov[11])
    	self.set_trans_matrix()

    def set_state_covariance(self, init_cov):
        self.root_p = init_cov[0]
        self.root_v = init_cov[1]
        self.root_l_t = self.root_r_t = init_cov[2]
        self.root_l_w = self.root_r_w = init_cov[3]
        self.hip_l_t = self.hip_r_t = init_cov[4]
        self.hip_l_w = self.hip_r_w = init_cov[5]
        self.knee_l_t = self.knee_r_t = init_cov[6]
        self.knee_l_w = self.knee_r_w = init_cov[7]
        self.ankle_l_t = self.ankle_r_t = init_cov[8]
        self.ankle_l_w = self.ankle_r_w = init_cov[9]
        self.link_length = 1e-9
        self.state_cov_list = [self.root_p,self.root_v,self.root_l_t,self.root_l_w,self.hip_l_t,self.hip_l_w,self.knee_l_t,self.knee_l_w,self.ankle_l_t,self.ankle_l_w,self.root_r_t,self.root_r_w,self.hip_r_t,self.hip_r_w,self.knee_r_t,self.knee_r_w,self.ankle_r_t,self.ankle_r_w,self.link_length]

    def set_state_dim(self):
        self.root_dim = 3
        self.root_l_dim =  self.root_r_dim = 3
        self.hip_l_dim = self.hip_r_dim = 3
        self.knee_l_dim = self.knee_r_dim = 1
        self.ankle_l_dim = self.ankle_r_dim = 2
        self.link_length = 8
        self.state_cov_dim_list = [self.root_dim,self.root_dim,self.root_l_dim,self.root_l_dim,self.hip_l_dim,self.hip_l_dim,self.knee_l_dim,self.knee_l_dim,self.ankle_l_dim,self.ankle_l_dim,self.root_r_dim,self.root_r_dim,self.hip_r_dim,self.hip_r_dim,self.knee_r_dim,self.knee_r_dim,self.ankle_r_dim,self.ankle_r_dim,self.link_length]
        self.state_cov_total_dim = sum(self.state_cov_dim_list)

    def gen_trans_covariance(self):
        tmp = np.eye(self.state_cov_total_dim)

        idx = 0
        for i in range(len(self.state_cov_list)):
            for j in range(self.state_cov_dim_list[i]):
                tmp[idx+j][idx+j] = self.state_cov_list[i]
            idx = idx + self.state_cov_dim_list[i]

        self.trans_cov = tmp

    def set_trans_covariance(self, init_cov):
        self.set_state_covariance(init_cov)
        self.set_state_dim()
        self.gen_trans_covariance()

    def set_obs_covariance(self, init_obs_cov_factor):
        self.obs_cov_dim = 27
        self.obs_cov_factor = init_obs_cov_factor
        self.obs_cov = np.eye(self.obs_cov_dim)*self.obs_cov_factor

    def set_mean(self, init_mean):
        self.mean = init_mean

    def set_init_trans_cov(self, t_factor):
    	self.init_trans_cov_factor = t_factor
    	self.init_trans_cov = self.trans_cov * self.init_trans_cov_factor

    def set_trans_matrix(self):
        self.fps = 10
        self.is_velocity = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        tmp = np.eye(self.state_cov_total_dim)

        idx = 0
        for i in range(len(self.state_cov_list)):
        	if self.is_velocity[i] == 1:
	        	for j in range(self.state_cov_dim_list[i]):
	        		tmp[idx+j-self.state_cov_dim_list[i]][idx+j] = 1/self.fps
        	idx = idx + self.state_cov_dim_list[i]

        self.trans_matrx = tmp