import sys
sys.path.append('./code/')

from read_data import *
from skeleton import Skeleton
from calibration import Calibration
from IJU_filter import IJU_Filter_Controler
from MJU_filter import MJU_Filter_Controler
from canvas import Canvas
from regression import *

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

@check_time
def test(filename, test_num, model, isplot=True):
	data = read_data_skeleton(filename)
	skeletons = []
	for d in data:
		skeletons.append(Skeleton(d))

	cbr_num = 50
	calibration = Calibration(skeletons[0:cbr_num+1])
	init_mean = calibration.get_init_mean()
	init_cov = [1e-6, 1e-4, 1e-6, 1e-4, 1e-6, 1e-1, 1e-6, 1e-1, 1e-6, 1e-4, 1e-4, 100]
	if model == 'IJU':
		ukf = IJU_Filter_Controler(init_mean, init_cov)
	elif model == 'MJU':
		ukf = MJU_Filter_Controler(init_mean, init_cov)
	else:
		print(model, "is not exist model name")

	canvas = Canvas()
	skeletons = skeletons[cbr_num:]

	ground_data = np.array([])
	estimate_data = np.array([])
	test_num = min(len(skeletons), test_num)
	for i in range(test_num):
		ret = ukf.update(skeletons[i].get_measurement())
		ground_data = np.append(ground_data, np.array(skeletons[i].get_lower_measurement()).reshape(-1))
		estimate_data = np.append(estimate_data, np.array(ret).reshape(-1))
		if not isplot:
			continue
		try:
			canvas.draw(ground_data[i], estimate_data[i])
			time.sleep(1)
		except KeyboardInterrupt:
			break

	mse_ret = mean_squared_error(ground_data, estimate_data, test_num)
	print(mse_ret)

@check_time
def test_bruth_forth(filename, test_num, model):
	data = read_data_skeleton(filename)
	skeletons = []
	for d in data:
		skeletons.append(Skeleton(d))

	canvas = Canvas()
	cbr_num = 50
	calibration = Calibration(skeletons[0:cbr_num+1])
	init_mean = calibration.get_init_mean()
	mse_min = 1e9
	mes_min_cov = []
	mes_min_v = []
	skeletons = skeletons[cbr_num:]
	mes_min_ground = []
	mes_min_estimate = []
#	init_cov = [pos_cov, velo_cov_1, pos_cov, velo_cov_1, pos_cov, velo_cov_2, pos_cov, velo_cov_2, pos_cov, velo_cov_1, 1e-4, 100]
	pos_cov_list = [1e-8, 1e-6, 20]
	velo_cov_list = [1e-9, 1e-8, 5]
	velo_cov_2_list = [1e-9, 2e-8, 5]

	for pos_cov in np.linspace(pos_cov_list[0], pos_cov_list[1], num=pos_cov_list[2]):
		for velo_cov_1 in np.linspace(velo_cov_list[0], velo_cov_list[1], num=velo_cov_list[2]):
			for velo_cov_2 in np.linspace(velo_cov_2_list[0], velo_cov_2_list[1], num=velo_cov_2_list[3]):
				init_cov = [pos_cov, velo_cov_1, pos_cov, velo_cov_1, pos_cov, velo_cov_2, pos_cov, velo_cov_2, pos_cov, velo_cov_1, 1e-4, 100]
				init_min_v = [pos_cov, velo_cov_1, velo_cov_2]
				if model == 'IJU':
					ukf = IJU_Filter_Controler(init_mean, init_cov)
				elif model == 'MJU':
					ukf = MJU_Filter_Controler(init_mean, init_cov)
				else:
					print(model, "is not exist model name")

				ground_data = []
				estimate_data = []
				draw_ground_data = []
				draw_estimate_data = []

				test_num = min(len(skeletons), test_num)
				for i in range(test_num):
					ret = ukf.update(skeletons[i].get_measurement())
					ground_data = np.append(ground_data, np.array(skeletons[i].get_lower_measurement()).reshape(-1))
					estimate_data = np.append(estimate_data, np.array(ret).reshape(-1))
					draw_ground_data.append(skeletons[i].get_measurement())
					draw_estimate_data.append(ret)

				mse_ret = mean_squared_error(ground_data, estimate_data, test_num) 
				if mse_ret < mse_min:
					mse_min = mse_ret
					mes_min_cov = init_cov
					mes_min_v = init_min_v
					mes_min_ground = draw_ground_data
					mes_min_estimate = draw_estimate_data
		print(pos_cov, mes_min_v)

	print(mes_min_v)
	for i in range(10):
		canvas.draw(mes_min_ground[i], mes_min_estimate[i])

if __name__ == "__main__":
	test("input_stand.txt", 50, 'IJU', False)
	# test_bruth_forth("input_stand.txt", 50, 'IJU')
