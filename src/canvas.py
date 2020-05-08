import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import *

from mpl_toolkits.mplot3d import Axes3D

import time
from functools import wraps

def check_time(function):
	@wraps(function)
	def measure(*args, **kwargs):
		start_time = time.time()
		result = function(*args, **kwargs)
		end_time = time.time()
		print(f"@check_time: {function.__name__} took {end_time - start_time}")
		return result

	return measure

class Canvas:
	def __init__(self):
		self.graph = self.skeleton_graph()
		self.quivers = {}

	def skeleton_graph(self):
		graph = {}
		graph[0] = [1, 18, 22]
		graph[1] = [2]
		graph[2] = [4, 11, 3]
		# left hand
		graph[4] = [5]
		graph[5] = [6]
		graph[6] = [7]
		graph[7] = [8,10]
		graph[8] = [9]
		# right hand
		graph[11] = [12]
		graph[12] = [13]
		graph[13] = [14]
		graph[14] = [15, 17]
		graph[15] = [16]
		# left foot
		graph[18] = [19]
		graph[19] = [20]
		graph[20] = [21]
		# right foot
		graph[22] = [23]
		graph[23] = [24]
		graph[24] = [25]
		# head
		graph[3] = [26]
		graph[26] = [27]
		graph[27] = [28, 30]
		graph[28] = [29]
		graph[30] = [31]
		return graph

	def make_dic_data(self, in_data):
		data = {}
		for i in range(len(in_data.joints)):
			data[i] = in_data.joints[i].pos
		return data

	def draw_skeleton_vec(self, here, vec_color, data):
		if here in self.graph:
			for nx in self.graph[here]:
				if nx in data:
					self.quivers[(here,nx,vec_color)] = self.ax.quiver(data[here][0],
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

	def draw_skeleton(self, in_data, color, label):
		data = self.make_dic_data(in_data)
		self.draw_skeleton_vec(0, color, data)
		self.set_plot_label(color, label, data)

	def update_skeleton_vec(self, here, vec_color, data):
		if here in self.graph:
			for nx in self.graph[here]:
				if nx in data:
					self.quivers[(here,nx,vec_color)].remove()
					self.quivers[(here,nx,vec_color)] = self.ax.quiver(data[here][0],
					 			   data[here][1],
					 			   data[here][2],
					 			   data[nx][0] - data[here][0],
					 			   data[nx][1] - data[here][1],
					 			   data[nx][2] - data[here][2],
					 			   color=vec_color
					)
					self.update_skeleton_vec(nx, vec_color, data)

	def update_skeleton(self, in_data, color):
		data = self.make_dic_data(in_data)
		self.update_skeleton_vec(0, color, data)

	def draw_3D_plot(self, ground, estimate, idx):
		self.flg = plt.figure()
		self.ax = self.flg.add_subplot(111, projection='3d')
		self.title = self.ax.set_title('3D Test {}'.format(idx))
		self.draw_skeleton(ground, 'red', 'ground')
		self.draw_skeleton(estimate, 'blue', 'estimate')
		self.ax.legend()
		self.ax.set_xlabel('X Label')
		self.ax.set_ylabel('Y Label')
		self.ax.set_zlabel('Z Label')
		plt.show()

	def update_3D_plot(self, num, ground_data, estimate_data):
		self.update_skeleton(ground_data[num], 'red')
		self.update_skeleton(estimate_data[num], 'blue')
		self.title.set_text('3D Test {}'.format(num))
		self.ax.legend()

	@check_time
	def skeleton_3D_animation_save(self, ground_data, estimate_data, interval, save_img):
		test_num = int(len(ground_data))
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, projection='3d')
		self.ax.view_init(-75,-90)
		# plt.subplots_adjust(0.12,0.12,0.90,0.90,0.20,0.20)
		self.draw_skeleton(ground_data[0], 'red', 'ground')
		self.draw_skeleton(estimate_data[0], 'blue', 'estimate')
		self.title = self.ax.set_title('3D Test')
		self.ax.set_xlabel('X Label')
		self.ax.set_ylabel('Y Label')
		self.ax.set_zlabel('Z Label')
		ani = animation.FuncAnimation(self.fig, self.update_3D_plot, test_num, fargs=(ground_data, estimate_data), interval=interval, repeat=False)
		# plt.show()
		ani.save(save_img+"movie.gif", writer="imagemagick")

	def skeleton_3D_plot(self, ground_data, estimate_data):
		for i in range(len(ground_data)):
			try:
				self.draw_3D_plot(ground_data[i], estimate_data[i], i)
			except KeyboardInterrupt:
				break

	def pyplot_skeleton_point(self, ground_data, estimate_data, word_name, img_name, i):
		idx = range(len(ground_data[0]))
		fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
		axs[0][0].plot(idx, ground_data[0], '-', color='red')
		axs[0][0].plot(idx, estimate_data[0], '-', color='blue')
		axs[0][0].set_ylabel('position(mm)')
		axs[0][0].set_title('X position')
		axs[0][1].plot(idx, ground_data[1], '-', color='red')
		axs[0][1].plot(idx, estimate_data[1], '-', color='blue')
		axs[0][1].set_title('Y position')
		axs[0][2].plot(idx, ground_data[2], '-', color='red', label='ground')
		axs[0][2].plot(idx, estimate_data[2], '-', color='blue', label='estimate')
		axs[0][2].legend(loc="upper right")
		axs[0][2].set_title('Z position {}'.format(word_name))
		axs[1][0].plot(idx, estimate_data[0] - ground_data[0], '-', color='red')
		axs[1][0].set_xlabel('frame')
		axs[1][0].set_ylabel('error(mm)')
		axs[1][1].plot(idx, estimate_data[1] - ground_data[1], '-', color='red')
		axs[1][1].set_xlabel('frame')
		axs[1][2].plot(idx, estimate_data[2] - ground_data[2], '-', color='red')
		axs[1][2].set_xlabel('frame')
		fig.savefig(img_name+str(i)+'_'+word_name+'.png')
		plt.close(fig)

	def get_point_data(self, data, test_num):
		ret = []
		for j in range(data[0].sk_num):
			for k in range(3):
				for i in range(test_num):
					ret.append(data[i].joints[j].pos[k])
		return np.array(ret).reshape(data[0].sk_num, 3, test_num)

	def get_point_plot_data(self, ground_data, estimate_data, test_num):
		return self.get_point_data(ground_data, test_num), self.get_point_data(estimate_data, test_num)

	@check_time
	def skeleton_point_plot(self, ground_data, estimate_data, img_name):
		test_num = int(len(ground_data))
		ground_point_data, estimate_point_data = self.get_point_plot_data(ground_data, estimate_data, test_num)
		word_name = ["PELVIS","SPINE_NAVAL","SPINE_CHEST","NECK","CLAVICLE_LEFT","SHOULDER_LEFT","ELBOW_LEFT","WRIST_LEFT","HAND_LEFT","HANDTIP_LEFT","THUMB_LEFT","CLAVICLE_RIGHT","SHOULDER_RIGHT","ELBOW_RIGHT","WRIST_RIGHT","HAND_RIGHT","HANDTIP_RIGHT","THUMB_RIGHT","HIP_LEFT","KNEE_LEFT","ANKLE_LEFT","FOOT_LEFT","HIP_RIGHT","KNEE_RIGHT","ANKLE_RIGHT","FOOT_RIGHT","HEAD","NOSE","EYE_LEFT","EAR_LEFT","EYE_RIGHT","EAR_RIGHT"]
		for i in range(ground_point_data.shape[0]):
			self.pyplot_skeleton_point(ground_point_data[i], estimate_point_data[i], word_name[i], img_name, i)

	def get_length_data(self, data, test_num):
		ret = []
		for j in range(len(data[0].joint_to_joints)):
			for i in range(test_num):
				ret.append(data[i].joint_to_joints[j])
		return np.array(ret).reshape(len(data[0].joint_to_joints), test_num)

	def get_length_plot_data(self, ground_data, estimate_data, test_num):
		return self.get_length_data(ground_data, test_num), self.get_length_data(estimate_data, test_num)

	def pyplot_skeleton_length(self, ground_data, estimate_data, word_name, img_name, i):
		idx = range(len(ground_data))
		fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
		axs[0].plot(idx, ground_data, '-', color='red', label='ground')
		axs[0].plot(idx, estimate_data, '-', color='blue', label='estimate')
		axs[0].set_ylabel('length(mm)')
		axs[0].set_title('length {}'.format(word_name))
		axs[0].legend(loc="upper right")
		axs[1].plot(idx, estimate_data - ground_data, '-', color='red')
		axs[1].set_xlabel('frame')
		axs[1].set_ylabel('error(mm)')
		fig.savefig(img_name+str(i)+'_'+word_name+'.png')
		plt.close(fig)

	@check_time
	def skeleton_length_plot(self, ground_data, estimate_data, img_name):
		test_num = int(len(ground_data))
		ground_length_data, estimate_length_data = self.get_length_plot_data(ground_data, estimate_data, test_num)
		word_name = ["D_root_spine","D_spine_chest","D_l_chest_sh","D_l_sh_shc","D_l_shc_elbow","D_l_elbow_wrist","D_l_wrist_hand","D_l_hand_handtip","D_l_wrist_thumb","D_r_chest_sh","D_r_sh_shc","D_r_shc_elbow","D_r_elbow_wrist","D_r_wrist_hand","D_r_hand_handtip","D_r_wrist_thumb","D_chest_neck","D_neck_head","D_head_nose","D_l_nose_eye","D_l_eye_ear","D_r_nose_eye","D_r_eye_ear","D_l_root_hip","D_l_hip_knee","D_l_knee_ankle","D_l_ankle_foot","D_r_root_hip","D_r_hip_knee","D_r_knee_ankle","D_r_ankle_foot"]
		for i in range(ground_length_data.shape[0]):
			self.pyplot_skeleton_length(ground_length_data[i], estimate_length_data[i], word_name[i], img_name, i)