import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import *

from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from IPython.display import display
from IPython.display import clear_output

class Canvas:
	def __init__(self):
		self.skeleton_graph()
		self.lower_idx = [0, 12, 13, 14, 15, 16, 17, 18, 19]
		self.lower_l = [0, 12, 13, 14, 15]
		self.lower_r = [0, 16, 17, 18, 19]
		self.lower_l_s = [0, 1, 2, 3, 4]
		self.lower_r_s = [0, 5, 6, 7, 8]

	def skeleton_graph(self):
		self.graph = {}
		self.graph[0] = [1, 12, 16]
		self.graph[1] = [2]
		self.graph[2] = [3,4,8]
		self.graph[3] = [20]
		# left hand
		self.graph[4] = [5]
		self.graph[5] = [6]
		self.graph[6] = [7]
		# right hand
		self.graph[8] = [9]
		self.graph[9] = [10]
		self.graph[10] = [11]
		# left foot
		self.graph[12] = [13]
		self.graph[13] = [14]
		self.graph[14] = [15]
		# right foot
		self.graph[16] = [17]
		self.graph[17] = [18]
		self.graph[18] = [19]
		# head
		self.graph[20] = [21]
		self.graph[21] = [22, 24]
		self.graph[22] = [23]
		self.graph[24] = [25]

	def draw_skeleton_vec(self, here, vec_color, data):
		if here in self.graph:
			for nx in self.graph[here]:
				if nx in data:
					self.ax.quiver(data[here][0],
					 			   data[here][1],
					 			   data[here][2],
					 			   data[nx][0] - data[here][0],
					 			   data[nx][1] - data[here][1],
					 			   data[nx][2] - data[here][2],
					 			   color=vec_color
					)
					self.draw_skeleton_vec(nx, vec_color, data)

	def set_plot_label(self, vec_color, vec_label, data):
		self.ax.quiver(data[0][0], data[0][1], data[0][2], 0, 0, 0, color=vec_color, label=vec_label)

	def make_draw_ground_data(self, ground):
		data = {}
		for i in range(len(ground)):
			data[i] = [ground[i][0], ground[i][1], ground[i][2]]
		return data

	def make_draw_estimate_data(self, estimate):
		data = {}
		for i in range(int(len(estimate)/3)):
			data[self.lower_idx[i]] = [estimate[i*3],estimate[i*3+1],estimate[i*3+2]]
		return data

	def draw_ground(self, ground):
		data = self.make_draw_ground_data(ground)
		vec_color = 'green'
		vec_label = 'ground'
		self.draw_skeleton_vec(0, vec_color, data)
		self.set_plot_label(vec_color, vec_label, data)

	def draw_estimate(self, estimate):
		data = self.make_draw_estimate_data(estimate)
		vec_color = 'blue'
		vec_label = 'estimate'
		self.draw_skeleton_vec(0, vec_color, data)
		self.set_plot_label(vec_color, vec_label, data)

	def update_graph(self, num):
		self.draw_ground(self.ground_data[num])
		self.draw_estimate(self.estimate_data[num])

	def set_3D_plot(self, flag, idx=0):
		if flag == 0:
			self.flg = plt.figure()
			self.ax = self.flg.add_subplot(111, projection='3d')
			self.title = self.ax.set_title('3D Test {}'.format(idx))
		if flag == 1:
			self.ax.legend()
			self.ax.set_xlabel('X Label')
			self.ax.set_ylabel('Y Label')
			self.ax.set_zlabel('Z Label')
			plt.show()

	def draw_3D_plot(self, ground, estimate, idx):
		self.set_3D_plot(0, idx)
		self.draw_ground(ground)
		self.draw_estimate(estimate)
		self.set_3D_plot(1)

	def animate_3D_plot(self, ground, estimate, idx):
		self.set_3D_plot(0, idx)
		self.draw_ground(ground)
		self.draw_estimate(estimate)
		self.set_3D_plot(1)
		display(plt.gcf())
		clear_output(wait=True)

	def skeleton_3D_plot(self, ground_data, estimate_data, Ipython, test_num, sleep_t):
		isContinue = True
		test_num = min(test_num, len(ground_data))
		if not Ipython:
			for i in range(test_num):
				try:
					self.draw_3D_plot(ground_data[i], estimate_data[i], i)
				except KeyboardInterrupt:
					break
		else:
			for i in range(test_num):
				try:
					self.animate_3D_plot(ground_data[i], estimate_data[i], i)
					time.sleep(sleep_t)
				except KeyboardInterrupt:
					isContinue = False
					break
				if not isContinue:
					break

	def skeleton_point_plot(self, ground_data, estimate_data, test_num, img_save=False, img_name=""):
		test_num = min(test_num, len(ground_data))
		x_grounds = []
		y_grounds = []
		z_grounds = []
		x_estimates = []
		y_estimates = []
		z_estimates = []
		for i in range(int(len(estimate_data[0])/3)):
			x_ground = []
			y_ground = []
			z_ground = []
			x_estimate = []
			y_estimate = []
			z_estimate = []
			for j in range(test_num):
				x_ground.append(ground_data[j][self.lower_idx[i]][0])
				y_ground.append(ground_data[j][self.lower_idx[i]][1])
				z_ground.append(ground_data[j][self.lower_idx[i]][2])
				x_estimate.append(estimate_data[j][i*3+0])
				y_estimate.append(estimate_data[j][i*3+1])
				z_estimate.append(estimate_data[j][i*3+2])
			x_grounds.append(np.array(x_ground))
			y_grounds.append(np.array(y_ground))
			z_grounds.append(np.array(z_ground))
			x_estimates.append(np.array(x_estimate))
			y_estimates.append(np.array(y_estimate))
			z_estimates.append(np.array(z_estimate))
		idx = range(test_num)

		# while True:
		# 	print("===================================")
		# 	print("===============<Menu>==============")
		# 	print("root(0)		l_hip(1) 	l_knee(2)")
		# 	print("l_ankle(3) 	l_foot(4) 	r_hip(5)")
		# 	print("r_knee(6) 	r_ankle(7) 	r_foot(8)")
		# 	print("quit(9)")
		# 	print("===================================")

			# num = int(input("Select Number: "))

		word_name = {}
		word_name[0] = "root"
		word_name[1] = "left hip"
		word_name[2] = "left knee"
		word_name[3] = "left ankle"
		word_name[4] = "left foot"
		word_name[5] = "right hip"
		word_name[6] = "right knee"
		word_name[7] = "right ankle"
		word_name[8] = "right foot"

		for i in range(0,9):
			fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
			axs[0][0].plot(idx, x_grounds[i], '-', color='red')
			axs[0][0].plot(idx, x_estimates[i], '-', color='blue')
			axs[0][0].set_ylabel('position(mm)')
			axs[0][0].set_title('X position')
			axs[0][1].plot(idx, y_grounds[i], '-', color='red')
			axs[0][1].plot(idx, y_estimates[i], '-', color='blue')
			axs[0][1].set_title('Y position')
			axs[0][2].plot(idx, z_grounds[i], '-', color='red', label='ground')
			axs[0][2].plot(idx, z_estimates[i], '-', color='blue', label='estimate')
			axs[0][2].legend(loc="upper right")

			axs[0][2].set_title('Z position {}'.format(word_name[i]))

			axs[1][0].plot(idx, x_estimates[i] - x_grounds[i], '-', color='red')
			axs[1][0].set_xlabel('frame')
			axs[1][0].set_ylabel('error(mm)')
			axs[1][1].plot(idx, y_estimates[i] - y_grounds[i], '-', color='red')
			axs[1][1].set_xlabel('frame')
			axs[1][2].plot(idx, z_estimates[i] - z_grounds[i], '-', color='red')
			axs[1][2].set_xlabel('frame')

			if img_save:
				fig.savefig(img_name+'/'+str(i)+'_'+word_name[i]+'.png')
				plt.close(fig)

		if not img_save:
			plt.show()

	def cal_skeleton_length_ground(self, data):
		lower_len = []
		for i in range(len(self.lower_l)-1):
			lower_len.append(get_distance(data[self.lower_l[i]], data[self.lower_l[i+1]]))
		for i in range(len(self.lower_r)-1):
			lower_len.append(get_distance(data[self.lower_r[i]], data[self.lower_r[i+1]]))
		return lower_len

	def cal_skeleton_length_estimate(self, data):
		lower_len = []
		for i in range(len(self.lower_l_s)-1):
			sx = self.lower_l_s[i]*3
			ex = self.lower_l_s[i+1]*3
			lower_len.append(get_distance(data[sx:sx+3], data[ex:ex+3]))
		for i in range(len(self.lower_r_s)-1):
			sx = self.lower_r_s[i]*3
			ex = self.lower_r_s[i+1]*3
			lower_len.append(get_distance(data[sx:sx+3], data[ex:ex+3]))
		return lower_len

	def skeleton_length_plot(self, ground_data, estimate_data, test_num, img_save=False, img_name=""):
		test_num = min(test_num, len(ground_data))
		ground_l = {}
		estimate_l = {}
		for i in range(8):
			ground_l[i] = []
			estimate_l[i] = []

		for i in range(test_num):
			g_ret = self.cal_skeleton_length_ground(ground_data[i])
			e_ret = self.cal_skeleton_length_estimate(estimate_data[i])
			for j in range(8):
				ground_l[j].append(g_ret[j])
				estimate_l[j].append(e_ret[j])

		for i in range(8):
			ground_l[i] = np.array(ground_l[i])
			estimate_l[i] = np.array(estimate_l[i])

		idx = range(test_num)

		word_name = {}
		word_name[0] = "r_to_lh"
		word_name[1] = "lh_to_lk"
		word_name[2] = "lk_to_la"
		word_name[3] = "la_to_lf"
		word_name[4] = "r_to_rh"
		word_name[5] = "rh_to_rk"
		word_name[6] = "rk_to_ra"
		word_name[7] = "ra_to_rf"

		for i in range(8):
			fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
			axs[0].plot(idx, ground_l[i], '-', color='red', label='ground')
			axs[0].plot(idx, estimate_l[i], '-', color='blue', label='estimate')
			axs[0].set_ylabel('length(mm)')
			axs[0].set_title('length {}'.format(word_name[i]))
			axs[0].legend(loc="upper right")

			axs[1].plot(idx, estimate_l[i] - ground_l[i], '-', color='red')
			axs[1].set_xlabel('frame')
			axs[1].set_ylabel('error(mm)')
			if img_save:
				fig.savefig(img_name+'/'+str(i)+'_'+word_name[i]+'.png')
				plt.close(fig)

		if not img_save:
			plt.show()
