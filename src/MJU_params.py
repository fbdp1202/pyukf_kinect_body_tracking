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
        root_p = init_cov[0]
        root_v = init_cov[1]
        root_l_t = root_r_t = init_cov[2]
        root_l_w = root_r_w = init_cov[3]
        hip_l_t = hip_r_t = init_cov[4]
        hip_l_w = hip_r_w = init_cov[5]
        knee_l_t = knee_r_t = init_cov[6]
        knee_l_w = knee_r_w = init_cov[7]
        ankle_l_t = ankle_r_t = init_cov[8]
        ankle_l_w = ankle_r_w = init_cov[9]
        link_length = 1e-9
        self.state_cov_list = [root_p,root_v,root_l_t,root_l_w,hip_l_t,hip_l_w,knee_l_t,knee_l_w,ankle_l_t,ankle_l_w,root_r_t,root_r_w,hip_r_t,hip_r_w,knee_r_t,knee_r_w,ankle_r_t,ankle_r_w,link_length]

    def set_state_dim(self):
        root_dim = 3
        root_l_dim =  root_r_dim = 3
        hip_l_dim = hip_r_dim = 3
        knee_l_dim = knee_r_dim = 1
        ankle_l_dim = ankle_r_dim = 2
        link_length = 8
        self.state_cov_dim_list = [root_dim,root_dim,root_l_dim,root_l_dim,hip_l_dim,hip_l_dim,knee_l_dim,knee_l_dim,ankle_l_dim,ankle_l_dim,root_r_dim,root_r_dim,hip_r_dim,hip_r_dim,knee_r_dim,knee_r_dim,ankle_r_dim,ankle_r_dim,link_length]
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

'''
Upper body model
    state vector
    x = [
        root:               3D(6)   p(x,y,z), u(x,y,z),                             ;
        s_root:             3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        spine_naval:        3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        l_spine_chest:      3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        l_shoulder:         3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        l_shoulder_center:  3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        l_elbow:            1D(2)                       theta(y),       w(y),       ;
        r_spine_chest:      3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_shoulder:         3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_shoulder_center:  3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_elbow:            1D(2)                       theta(y),       w(y),       ;
        u_spine_chest:      3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        neck:               3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        L_root_spine:       1D(1)   L                                               ,
        L_spine_chest:      1D(1)   L                                               ,
        L_l_chest_sh:       1D(1)   L                                               ,
        L_l_sh_shc:         1D(1)   L                                               ,
        L_l_shc_elbow:      1D(1)   L                                               ,
        L_l_elbow_wrist:    1D(1)   L                                               ,
        L_r_chest_sh:       1D(1)   L                                               ,
        L_r_sh_shc:         1D(1)   L                                               ,
        L_r_shc_elbow:      1D(1)   L                                               ,
        L_r_elbow_wrist:    1D(1)   L                                               ,
        L_chest_neck:       1D(1)   L                                               ,
        L_neck_head:        1D(1)   L                                               ,
    ]
    total vector length(DP): 82

    measurement vector
    y = [
        root:               3D(3)       0
        spine_naval:        3D(3)       1
        spine_chest:        3D(3)       2
        l_shoulder:         3D(3)       4
        l_shoulder_center:  3D(3)       5
        l_elbow:            3D(3)       6
        l_wrist:            3D(3)       7
        r_shoulder:         3D(3)       8
        r_shoulder_center:  3D(3)       9
        r_elbow:            3D(3)       10
        r_wrist:            3D(3)       11
        neck:               3D(3)       3
        head:               3D(3)       20
    ]
    total vector length(MP): 39
'''

class MJU_Upper_Params:
    def __init__(self, init_mean, init_cov=[]):
        self.set_trans_covariance(init_cov[:18])
        self.set_obs_covariance(init_cov[19])
        self.set_mean(init_mean)
        self.set_init_trans_cov(init_cov[20])
        self.set_trans_matrix()

    def set_state_covariance(self, init_cov):
        root_p, root_v = init_cov[0:2]
        s_root_t, s_root_w = init_cov[2:4]
        spine_naval_t, spine_naval_w = init_cov[4:6]
        l_spine_chest_t, l_spine_chest_w = init_cov[6:8]
        l_shoulder_t, l_shoulder_w = init_cov[8:10]
        l_shoulder_center_t, l_shoulder_center_w = init_cov[10:12]
        l_elbow_t, l_elbow_w = init_cov[12:14]
        r_spine_chest_t, r_spine_chest_w = init_cov[6:8]
        r_shoulder_t, r_shoulder_w = init_cov[8:10]
        r_shoulder_center_t, r_shoulder_center_w = init_cov[10:12]
        r_elbow_t, r_elbow_w = init_cov[12:14]
        u_spine_chest_t, u_spine_chest_w = init_cov[14:16]
        neck_t, neck_w = init_cov[16:18]
        link_length = 1e-9
        self.state_cov_list = [root_p,root_v,s_root_t,s_root_w,spine_naval_t,spine_naval_w,l_spine_chest_t,l_spine_chest_w,l_shoulder_t,l_shoulder_w,l_shoulder_center_t,l_shoulder_center_w,l_elbow_t,l_elbow_w,r_spine_chest_t,r_spine_chest_w,r_shoulder_t,r_shoulder_w,r_shoulder_center_t,r_shoulder_center_w,r_elbow_t,r_elbow_w,u_spine_chest_t,u_spine_chest_w,neck_t,neck_w,link_length]

    def set_state_dim(self):
        root_dim = 3
        s_root_dim = 3
        spine_naval_dim = 3
        l_spine_chest_dim = 3
        l_shoulder_dim = 3
        l_shoulder_center_dim = 3
        l_elbow_dim = 1
        r_spine_chest_dim = 3
        r_shoulder_dim = 3
        r_shoulder_center_dim = 3
        r_elbow_dim = 1
        u_spine_chest_dim = 3
        neck_dim = 3
        link_length_dim = 12
        self.state_cov_dim_list = [root_dim,root_dim,s_root_dim,s_root_dim,spine_naval_dim,spine_naval_dim,l_spine_chest_dim,l_spine_chest_dim,l_shoulder_dim,l_shoulder_dim,l_shoulder_center_dim,l_shoulder_center_dim,l_elbow_dim,l_elbow_dim,r_spine_chest_dim,r_spine_chest_dim,r_shoulder_dim,r_shoulder_dim,r_shoulder_center_dim,r_shoulder_center_dim,r_elbow_dim,r_elbow_dim,u_spine_chest_dim,u_spine_chest_dim,neck_dim,neck_dim,link_length_dim]
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
        self.obs_cov_dim = 39
        self.obs_cov_factor = init_obs_cov_factor
        self.obs_cov = np.eye(self.obs_cov_dim)*self.obs_cov_factor

    def set_mean(self, init_mean):
        self.mean = init_mean

    def set_init_trans_cov(self, t_factor):
        self.init_trans_cov_factor = t_factor
        self.init_trans_cov = self.trans_cov * self.init_trans_cov_factor

    def set_trans_matrix(self):
        self.fps = 10
        self.is_velocity = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        tmp = np.eye(self.state_cov_total_dim)

        idx = 0
        for i in range(len(self.state_cov_list)):
            if self.is_velocity[i] == 1:
                for j in range(self.state_cov_dim_list[i]):
                    tmp[idx+j-self.state_cov_dim_list[i]][idx+j] = 1/self.fps
            idx = idx + self.state_cov_dim_list[i]

        self.trans_matrx = tmp
