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

		D_l_root_hip: 	1D(1)	L												,
		D_l_hip_knee: 	1D(1)	L												,
		D_l_knee_ankle: 	1D(1)	L												,
		D_l_ankle_foot: 	1D(1)	L												,
		D_r_root_hip: 	1D(1)	L												,
		D_r_hip_knee: 	1D(1)	L												,
		D_r_knee_ankle: 	1D(1)	L												,
		D_r_ankle_foot: 	1D(1)	L												,
	]
    cal: 42 + 8
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
class ukf_Lower_Params:
    def __init__(self, init_mean, init_cov=[1e-6, 1e-4, 1e-6, 1e-4, 1e-6, 1e-1, 1e-6, 1e-1, 1e-6, 1e-4, 1e-9, 1e-4, 100]):
    	self.set_trans_covariance(init_cov[:11])
    	self.set_obs_covariance(init_cov[11])
    	self.set_mean(init_mean)
    	self.set_init_trans_cov(init_cov[12])
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
        link_length = init_cov[10]
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
        l_elbow:            3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        l_wrist_u:          3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        l_hand:             3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        l_wrist_d:          3D(6)                       theta(x,y,z),   w(x,y,z),   ;

        r_spine_chest:      3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_shoulder:         3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_shoulder_center:  3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_elbow:            3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_wrist_u:          3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_hand:             3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_wrist_d:          3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        u_spine_chest:      3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        neck:               3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        head:               3D(6)                       theta(x,y,z),   w(x,y,z),   ;

        l_nose:             3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        l_eye:              3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_nose:             3D(6)                       theta(x,y,z),   w(x,y,z),   ;
        r_eye:              3D(6)                       theta(x,y,z),   w(x,y,z),   ;

        D_root_spine:       1D(1)                                                   ,
        D_spine_chest:      1D(1)                                                   ,
        D_l_chest_sh:       1D(1)                                                   ,
        D_l_sh_shc:         1D(1)                                                   ,
        D_l_shc_elbow:      1D(1)                                                   ,
        D_l_elbow_wrist:    1D(1)                                                   ,
        D_l_wrist_hand      1D(1)                                                   ,
        D_l_hand_handtip:   1D(1)                                                   ,
        D_l_wrist_thumb:    1D(1)                                                   ,
        D_r_chest_sh:       1D(1)                                                   ,

        D_r_sh_shc:         1D(1)                                                   ,
        D_r_shc_elbow:      1D(1)                                                   ,
        D_r_elbow_wrist:    1D(1)                                                   ,
        D_r_wrist_hand:     1D(1)                                                   ,
        D_r_hand_handtip:   1D(1)                                                   ,
        D_r_wrist_thumb:    1D(1)                                                   ,
        D_chest_neck:       1D(1)                                                   ,
        D_neck_head:        1D(1)                                                   ,
        D_head_nose:        1D(1)                                                   ,
        D_l_nose_eye:       1D(1)                                                   ,

        D_l_eye_ear:        1D(1)                                                   ,
        D_r_nose_eye:       1D(1)                                                   ,
        D_r_eye_ear:        1D(1)                                                   ,
    ]
    cal: 24*6 - 8 + 23 = 144 + 23 = 167
    total vector length(DP): 167

    measurement vector
    y = [
        root:               3D(3)       0
        spine_naval:        3D(3)       1
        spine_chest:        3D(3)       2
        l_shoulder:         3D(3)       4
        l_shoulder_center:  3D(3)       5
        l_elbow:            3D(3)       6
        l_wrist:            3D(3)       7
        l_hand:             3D(3)       8
        l_handtip:          3D(3)       9
        l_thumb:            3D(3)       10

        r_shoulder:         3D(3)       11
        r_shoulder_center:  3D(3)       12
        r_elbow:            3D(3)       13
        r_wrist:            3D(3)       14
        r_hand:             3D(3)       15
        r_handtip:          3D(3)       16
        r_thumb:            3D(3)       17
        neck:               3D(3)       3
        head:               3D(3)       26
        nose:               3D(3)       27

        l_eye:              3D(3)       28
        l_ear:              3D(3)       29
        r_eye:              3D(3)       30
        r_ear:              3D(3)       31
    ]
    cal: 24 * 3 = 72
    total vector length(MP): 72
'''

class ukf_Upper_Params:
    def __init__(self, init_mean, init_cov=[]):
        self.set_trans_covariance(init_cov[:35])
        self.set_obs_covariance(init_cov[35])
        self.set_mean(init_mean)
        self.set_init_trans_cov(init_cov[36])
        self.set_trans_matrix()

    def set_state_covariance(self, init_cov):
        root_p,                 root_v              = init_cov[0:2]
        s_root_t,               s_root_w            = init_cov[2:4]
        spine_naval_t,          spine_naval_w       = init_cov[4:6]

        l_spine_chest_t,        l_spine_chest_w     = init_cov[6:8]
        l_shoulder_t,           l_shoulder_w        = init_cov[8:10]
        l_shoulder_center_t,    l_shoulder_center_w = init_cov[10:12]
        l_elbow_t,              l_elbow_w           = init_cov[12:14]
        l_wrist_u_t,            l_wrist_u_w         = init_cov[14:16]
        l_hand_t,               l_hand_w            = init_cov[16:18]
        l_wrist_d_t,            l_wrist_d_w         = init_cov[18:20]

        r_spine_chest_t,        r_spine_chest_w     = init_cov[6:8]
        r_shoulder_t,           r_shoulder_w        = init_cov[8:10]
        r_shoulder_center_t,    r_shoulder_center_w = init_cov[10:12]
        r_elbow_t,              r_elbow_w           = init_cov[12:14]
        r_wrist_u_t,            r_wrist_u_w         = init_cov[14:16]
        r_hand_t,               r_hand_w            = init_cov[16:18]
        r_wrist_d_t,            r_wrist_d_w         = init_cov[18:20]

        u_spine_chest_t,        u_spine_chest_w     = init_cov[20:22]
        neck_t,                 neck_w              = init_cov[22:24]
        head_t,                 head_w              = init_cov[24:26]
        l_nose_t,               l_nose_w            = init_cov[26:28]
        l_eye_t,                l_eye_w             = init_cov[28:30]
        r_nose_t,               r_nose_w            = init_cov[30:32]
        r_eye_t,                r_eye_w             = init_cov[32:34]

        link_length                                 = init_cov[34]

        self.state_cov_list = [root_p,root_v,s_root_t,s_root_w,spine_naval_t,spine_naval_w,l_spine_chest_t,l_spine_chest_w,l_shoulder_t,l_shoulder_w,l_shoulder_center_t,l_shoulder_center_w,l_elbow_t,l_elbow_w,l_wrist_u_t,l_wrist_u_w,l_hand_t,l_hand_w,l_wrist_d_t,l_wrist_d_w,r_spine_chest_t,r_spine_chest_w,r_shoulder_t,r_shoulder_w,r_shoulder_center_t,r_shoulder_center_w,r_elbow_t,r_elbow_w,r_wrist_u_t,r_wrist_u_w,r_hand_t,r_hand_w,r_wrist_d_t,r_wrist_d_w,u_spine_chest_t,u_spine_chest_w,neck_t,neck_w,head_t,head_w,l_nose_t,l_nose_w,l_eye_t,l_eye_w,r_nose_t,r_nose_w,r_eye_t,r_eye_w,link_length]

    def set_state_dim(self):
        root_dim = 3
        s_root_dim = 3
        spine_naval_dim = 3

        l_spine_chest_dim = 3
        l_shoulder_dim = 3
        l_shoulder_center_dim = 3
        l_elbow_dim = 3
        l_wrist_u_dim = 3
        l_hand_dim = 3
        l_wrist_d_dim = 3

        r_spine_chest_dim = 3
        r_shoulder_dim = 3
        r_shoulder_center_dim = 3
        r_elbow_dim = 3
        r_wrist_u_dim = 3
        r_hand_dim = 3
        r_wrist_d_dim = 3

        u_spine_chest_dim = 3
        neck_dim = 3
        head_dim = 3
        l_nose_dim = 3
        l_eye_dim = 3
        r_nose_dim = 3
        r_eye_dim = 3

        link_length_dim = 23

        self.state_cov_dim_list = [root_dim,root_dim,s_root_dim,s_root_dim,spine_naval_dim,spine_naval_dim,l_spine_chest_dim,l_spine_chest_dim,l_shoulder_dim,l_shoulder_dim,l_shoulder_center_dim,l_shoulder_center_dim,l_elbow_dim,l_elbow_dim,l_wrist_u_dim,l_wrist_u_dim,l_hand_dim,l_hand_dim,l_wrist_d_dim,l_wrist_d_dim,r_spine_chest_dim,r_spine_chest_dim,r_shoulder_dim,r_shoulder_dim,r_shoulder_center_dim,r_shoulder_center_dim,r_elbow_dim,r_elbow_dim,r_wrist_u_dim,r_wrist_u_dim,r_hand_dim,r_hand_dim,r_wrist_d_dim,r_wrist_d_dim,u_spine_chest_dim,u_spine_chest_dim,neck_dim,neck_dim,head_dim,head_dim,l_nose_dim,l_nose_dim,l_eye_dim,l_eye_dim,r_nose_dim,r_nose_dim,r_eye_dim,r_eye_dim,link_length_dim]
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
        self.obs_cov_dim = 72
        self.obs_cov_factor = init_obs_cov_factor
        self.obs_cov = np.eye(self.obs_cov_dim)*self.obs_cov_factor

    def set_mean(self, init_mean):
        self.mean = init_mean

    def set_init_trans_cov(self, t_factor):
        self.init_trans_cov_factor = t_factor
        self.init_trans_cov = self.trans_cov * self.init_trans_cov_factor

    def set_trans_matrix(self):
        self.fps = 10
        self.is_velocity = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        tmp = np.eye(self.state_cov_total_dim)

        idx = 0
        for i in range(len(self.state_cov_list)):
            if self.is_velocity[i] == 1:
                for j in range(self.state_cov_dim_list[i]):
                    tmp[idx+j-self.state_cov_dim_list[i]][idx+j] = 1/self.fps
            idx = idx + self.state_cov_dim_list[i]

        self.trans_matrx = tmp
