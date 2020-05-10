import sys
import os
sys.path.append('./code/')

from skeleton import Skeleton
from read_data import *
from calibration import Calibration
from ukf_filter import ukf_Filter_Controler
from canvas import Canvas
from regression import *

import time
from functools import wraps

import os

def check_time(function):
	@wraps(function)
	def measure(*args, **kwargs):
		start_time = time.time()
		result = function(*args, **kwargs)
		end_time = time.time()
		print(f"@check_time: {function.__name__} took {end_time - start_time}")
		return result

	return measure

def get_dir_name(dir):
	dir_list = []
	for name in os.listdir(dir):
		path = dir + '/' + name
		if not os.path.isfile(path):
			dir_list.append(name)
	return dir_list

def scan_dir(dir):
	dir_list = []
	for name in os.listdir(dir):
		path = dir + '/' + name
		if os.path.isfile(path):
			dir_list.append(path)
	return dir_list

@check_time
def merge_skeleton_data(folder_name):
	save_file_name = folder_name + '.txt' 
	dir_list = scan_dir(folder_name)

	wf = open(save_file_name, 'w')
	for filename in dir_list:
		f = open(filename, 'r')
		line = f.readline()
		wf.write(line)
	wf.close()
	return save_file_name

@check_time
def init_simul(filename, test_num, cbr_num=50, div_step=1):
	data = read_data_skeleton(filename)
	# test_num, data = interval_compasation(data, test_num, div_step)
	test_num = min(test_num, len(data))

	skeletons = []
	for i in range(test_num):
		skeletons.append(Skeleton(data[i]))

	cbr_num = min(test_num, cbr_num)
	cal_skeletons = []
	for i in range(cbr_num):
		cal_skeletons.append(skeletons[i*div_step])

	calibration = Calibration(cal_skeletons)
	lower_init_mean, upper_init_mean = calibration.get_init_mean(0, filename)

	return skeletons, lower_init_mean, upper_init_mean, test_num

@check_time
def make_filter(lower_init_mean,  lower_init_cov, upper_init_mean, upper_init_cov, model):
	flt = None
	if model == 'ukf':
		flt = ukf_Filter_Controler(lower_init_mean, lower_init_cov, upper_init_mean, upper_init_cov)
	else:
		print(model, "is not exist model name")
	return flt

@check_time
def run_ukf(ukf, skeletons, test_num):
	original_data = []
	estimate_data = []
	estimate_state = []

	test_num = min(len(skeletons), test_num)
	print("total test is {}".format(test_num))
	print("test_num:", end=' ')
	for i in range(test_num):
		curr_input = skeletons[i].get_measurement()
		original_data.append(curr_input)
		state, data = ukf.update(curr_input)
		estimate_data.append(data)
		estimate_state.append(state)
		if i % 10 == 0:
			print(i, end=' ')
	print('')

	return original_data, estimate_data, estimate_state

def make_folder(folder_name):
	if not os.path.isdir(folder_name):
		os.mkdir(folder_name)
	return folder_name

def get_save_skeleton_data_folder_name(person_name, pos_mode, model):
	folder_name = make_folder('result')
	folder_name = make_folder(folder_name + '/' + person_name)
	folder_name = make_folder(folder_name + '/' + pos_mode)
	folder_name = make_folder(folder_name + '/' + model)
	return folder_name + '/'

def save_sk_data_to_csv(folder_name, filename, data):
	filename = folder_name + filename
	f = open(filename, "w", encoding="UTF-8")
	for i in range(len(data)):
		for j in range(len(data[i])):
			for k in range(3):
				f.write(str(data[i][j][k]))
				if j == (len(data[i])-1) and k == 2:
					f.write('\n')
				else:
					f.write(',')

def save_sk_state_to_csv(folder_name, filename, data):
	filename = folder_name + filename
	f = open(filename, 'w', encoding="UTF-8")
	for i in range(len(data)):
		for j in range(len(data[i])):
			f.write(str(data[i][j]))
			if j == (len(data[i])-1):
				f.write('\n')
			else:
				f.write(',')

@check_time
def save_skeleton_data_to_csv(person_name, pos_mode, original_data, estimate_data, estimate_state, model):
	csv_folder_name = get_save_skeleton_data_folder_name(person_name, pos_mode, model)
	save_sk_data_to_csv(csv_folder_name, 'original_data.csv', original_data)
	save_sk_data_to_csv(csv_folder_name, 'estimate_data.csv', estimate_data)
	save_sk_state_to_csv(csv_folder_name, 'estimate_state.csv', estimate_state)

def read_csv(filename):
	data = []
	with open(filename, 'r') as reader:
		for line in reader:
			fields = line.split(',')
			fields[len(fields)-1] = fields[len(fields)-1].replace('\n', '')
			for i in range(len(fields)):
				data.append(float(fields[i]))

	data = np.array(data).reshape((int)(len(data)/32/3), 32, 3)
	skeletons = []
	for d in data:
		skeletons.append(Skeleton(d))
	return skeletons

@check_time
def read_skeleton_data_from_csv(person_name, pos_mode, model):
	csv_folder_name = get_save_skeleton_data_folder_name(person_name, pos_mode, model)
	original_data = read_csv(csv_folder_name + 'original_data.csv')
	estimate_data = read_csv(csv_folder_name + 'estimate_data.csv')
	return original_data, estimate_data

def get_save_image_file_name(person_name, pos_mode, model, plot_mode):
	folder_name = make_folder('result')
	folder_name = make_folder(folder_name + '/' + person_name)
	folder_name = make_folder(folder_name + '/' + pos_mode)
	folder_name = make_folder(folder_name + '/' + model)
	folder_name = make_folder(folder_name + '/' + plot_mode)
	return folder_name + '/'

@check_time
def skeleton_draw(person_name, pos_mode, model, original_data, estimate_data, sleep_t=100):
	canvas = Canvas()
	img_name_point = get_save_image_file_name(person_name, pos_mode, model, 'point')
	img_name_length = get_save_image_file_name(person_name, pos_mode, model, 'length')
	img_name_3D = get_save_image_file_name(person_name, pos_mode, model, 'plot_3D')
	# 	canvas.skeleton_3D_plot(original_data, estimate_data)

	canvas.skeleton_3D_animation_save(original_data, estimate_data, sleep_t, img_name_3D)
	canvas.skeleton_point_plot(original_data, estimate_data, img_name_point)
	canvas.skeleton_length_plot(original_data, estimate_data, img_name_length)

def set_lower_init_cov(value_cov=1e-6, velo_cov_0=1e-4, velo_cov_1=1e-2, len_cov=1e-10, obs_cov_factor=1e-4, trans_factor=100):
	return [value_cov, velo_cov_0,value_cov, velo_cov_0,value_cov, velo_cov_1,value_cov, velo_cov_1,value_cov, velo_cov_0, len_cov,obs_cov_factor, trans_factor]

def set_upper_init_cov(value_cov=1e-6, velo_cov=1e-4, len_cov=1e-10, obs_cov_factor=1e-4, trans_factor=100):
	return [value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,value_cov,velo_cov,len_cov,obs_cov_factor,trans_factor]

@check_time
def simulation_ukf(filename, test_num, cbr_num, model):
	skeletons, lower_init_mean, upper_init_mean, test_num = init_simul(filename, test_num, cbr_num)

	lower_init_cov = set_lower_init_cov()
	upper_init_cov = set_upper_init_cov()
	flt = make_filter(lower_init_mean, lower_init_cov, upper_init_mean, upper_init_cov, model)

	original_data, estimate_data, estimate_state = run_ukf(flt, skeletons, test_num)
	return original_data, estimate_data, estimate_state
