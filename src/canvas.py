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

	def draw_skeleton_vec(self, here, vec_color):
		if here in self.graph:
			for nx in self.graph[here]:
				if nx in self.data:
					self.ax.quiver(self.data[here][0],
					 			   self.data[here][1],
					 			   self.data[here][2],
					 			   self.data[nx][0] - self.data[here][0],
					 			   self.data[nx][1] - self.data[here][1],
					 			   self.data[nx][2] - self.data[here][2],
					 			   color=vec_color
					)
					self.draw_skeleton_vec(nx, vec_color)

	def set_plot_label(self, vec_color, vec_label):
		self.ax.quiver(self.data[0][0], self.data[0][1], self.data[0][2], 0, 0, 0, color=vec_color, label=vec_label)


	def draw_ground(self,ground):
		self.data = {}
		for i in range(len(ground)):
			self.data[i] = [float(ground[i][0]), float(ground[i][1]), float(ground[i][2])]
		vec_color = 'green'
		vec_label = 'ground'
		self.draw_skeleton_vec(0, vec_color)
		self.set_plot_label(vec_color, vec_label)

	def draw_estimate(self,estimate):
		self.data = {}
		for i in range(int(len(estimate)/3)):
			self.data[self.lower_idx[i]] = [estimate[i*3],estimate[i*3+1],estimate[i*3+2]]
		vec_color = 'blue'
		vec_label = 'estimate'
		self.draw_skeleton_vec(0, vec_color)
		self.set_plot_label(vec_color, vec_label)

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