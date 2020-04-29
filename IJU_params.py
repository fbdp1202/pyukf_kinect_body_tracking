import numpy as np

'''
Lower body model
	state vector = [
    	[root:    p(x,y,z),      u(x,y,z)    ]
        [root_l:  theta(x,y,z),  w(x,y,z), L ]
        [root_r:  theta(x,y,z),  w(x,y,z), L ]
        [l-hip:   theta(x,y,z),  w(x,y,z), L ]
        [l-knee:  theta(y),      w(y),     L ]
        [l-ankle: theta(y,z),    w(y,z),   L ]
        [r-hip:   theta(x,y,z),  w(x,y,z), L ]
        [r-knee:  theta(y),      w(y),     L ]
        [r-ankle: theta(y,z),    w(y,z),   L ]
	]

	measurement vector
	y = [
		[x_root: 		3D(3) ]
		[x_l_hip: 		3D(3) ]
		[x_l-knee:		3D(3) ]
		[x_l-ankle:		3D(3) ]
		[x_l-foot:		3D(3) ]
		[x_r-hip:		3D(3) ]
		[x_r-knee:		3D(3) ]
		[x_r-ankle:		3D(3) ]
		[x_r-foot:		3D(3) ]
	]
	total vector length(MP): 27
'''
class IJU_Lower_Params:
    def __init__(self):
        pass

    def __init__(self, init_mean, init_cov):
        self.lower_point = [0, 12, 13, 14, 15, 16, 17, 18, 19]
        self.set_trans_covariance(init_cov[:10])
        self.set_obs_covariance(init_cov[10])
        self.set_mean(init_mean)
        self.set_init_trans_cov(init_cov[11])
        self.set_trans_matrix()
        self.set_graph()


    def gen_cov(self, data, dim):
        total_dim = dim * 2
        if data[2] != 0:
            total_dim = total_dim + 1
        tmp = np.eye(total_dim)
        for i in range(dim):
            tmp[i][i] = data[0]
            tmp[i+dim][i+dim] = data[1]
        if data[2] != 0:
            tmp[dim*2][dim*2] = data[2]
        return tmp

    def set_state_covariance(self, init_cov):
        self.cov_value = {}                         # (pos, velocity, link_length)
        self.cov_value[0] = (init_cov[0], init_cov[1], 0)         #  root
        self.cov_value[1] = (init_cov[2], init_cov[3], 1e-9)      #  l_root
        self.cov_value[2] = (init_cov[4], init_cov[5], 1e-9)      #  l_hip
        self.cov_value[3] = (init_cov[6], init_cov[7], 1e-9)      #  l_knee
        self.cov_value[4] = (init_cov[8], init_cov[9], 1e-9)      #  l_ankle
        self.cov_value[5] = (init_cov[2], init_cov[3], 1e-9)      #  r_root
        self.cov_value[6] = (init_cov[4], init_cov[5], 1e-9)      #  r_hip
        self.cov_value[7] = (init_cov[6], init_cov[7], 1e-9)      #  r_knee
        self.cov_value[8] = (init_cov[8], init_cov[9], 1e-9)      #  r_ankle

    def set_state_dim(self):
        self.cov_dim = {}
        self.cov_dim[0] = 3         #  root
        self.cov_dim[1] = 3         #  l_root
        self.cov_dim[2] = 3         #  l_hip
        self.cov_dim[3] = 1         #  l_knee
        self.cov_dim[4] = 2         #  l_ankle
        self.cov_dim[5] = 3         #  r_root
        self.cov_dim[6] = 3         #  r_hip
        self.cov_dim[7] = 1         #  r_knee
        self.cov_dim[8] = 2         #  r_ankle

    def gen_trans_covariance(self):
        self.trans_cov = {}
        for i in range(9):
            self.trans_cov[self.lower_point[i]] = self.gen_cov(self.cov_value[i], self.cov_dim[i])

    def set_trans_covariance(self, init_cov):
        self.set_state_covariance(init_cov)
        self.set_state_dim()
        self.gen_trans_covariance()

    def set_obs_covariance(self, init_obs_cov_factor):
        self.obs_cov_dim = 27
        self.obs_cov_factor = init_obs_cov_factor
        self.obs_cov = {}
        for i in range(9):
            self.obs_cov[self.lower_point[i]] = np.eye(3)*self.obs_cov_factor

    def set_mean(self, init_mean):
        bias_dim = 0
        for i in range(9):
            bias_dim = bias_dim + self.cov_dim[i] * 2
        self.mean = {}
        self.mean[self.lower_point[0]] = init_mean[:self.cov_dim[0]*2]
        cur = self.cov_dim[0]*2
        for i in range(1,9):
            self.mean[self.lower_point[i]] = list(init_mean[cur:cur+self.cov_dim[i]*2]) + [init_mean[bias_dim+i-1]]
            cur = cur + self.cov_dim[i]*2

    def set_init_trans_cov(self, t_factor):
        self.init_trans_cov_factor = t_factor
        self.init_trans_cov = {}
        for i in range(9):
            self.init_trans_cov[self.lower_point[i]] = self.init_trans_cov_factor * self.trans_cov[self.lower_point[i]]

    def gen_trans_matrix(self, dim, idx):
        total_dim = dim * 2
        if idx != 0:
            total_dim = total_dim + 1
        tmp = np.eye(total_dim)
        for i in range(dim):
            tmp[i][dim+i] = 1/self.fps
        return tmp


    def set_trans_matrix(self):
        self.fps = 3
        self.trans_matrx = {}
        for i in range(9):
            self.trans_matrx[self.lower_point[i]] = self.gen_trans_matrix(self.cov_dim[i], i)

    def set_graph(self):
        self.graph = {}
        self.graph[0] = [12, 16]
        # left foot
        self.graph[12] = [13]
        self.graph[13] = [14]
        self.graph[14] = [15]
        # right foot
        self.graph[16] = [17]
        self.graph[17] = [18]
        self.graph[18] = [19]
