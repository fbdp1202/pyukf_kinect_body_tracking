import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from IPython.display import display
from IPython.display import clear_output

class Canvas:
	def __init__(self):
		self.skeleton_graph()
		self.lower_idx = [0, 12, 13, 14, 15, 16, 17, 18, 19]

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
		data = make_draw_estimate_data(estimate)
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

	def skeleton_3D_plot(ground_data, estimate_data, Ipython, test_num, sleep_t):
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

	def skeleton_point_plot(ground_data, estimate_data, test_num):
		test_num = min(test_num, len(ground_data))
		num = 0
		x_ground = []
		y_ground = []
		z_ground = []
		x_estimate = []
		y_estimate = []
		z_estimate = []
		idx = range(test_num)
		for i in range(test_num):
			x_ground.append(ground_data[i][num][0])
			y_ground.append(ground_data[i][num][1])
			z_ground.append(ground_data[i][num][2])
			x_estimate.append(ground_data[i][num*3+0])
			y_estimate.append(ground_data[i][num*3+1])
			z_estimate.append(ground_data[i][num*3+2])

		plt.subplot(1,3,1)
		plt.plot(idx, x_ground, color='red')
		plt.plot(idx, x_estimate, color='blue')
		plt.xlabel('frame')
		plt.ylabel('X position(mm)')
		plt.title('X position Display')

		plt.subplot(1,3,2)
		plt.plot(idx, y_ground, color='red')
		plt.plot(idx, y_estimate, color='blue')
		plt.xlabel('frame')
		plt.ylabel('Y position(mm)')
		plt.title('Y position Display')

		plt.subplot(1,3,3)
		plt.plot(idx, z_ground, color='red')
		plt.plot(idx, z_estimate, color='blue')
		plt.xlabel('frame')
		plt.ylabel('Z position(mm)')
		plt.title('Z position Display')
		plt.show()

