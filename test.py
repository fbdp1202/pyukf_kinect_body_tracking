import sys
sys.path.append('./src/')
from main import *

def test(filename, test_num, model, isplot=True, plot_mode='3D', Ipython=False, cbr_num=50, save_img=False):
	ground_data, estimate_data = simulation_ukf(filename, test_num, model, cbr_num)
	if isplot:
		skeleton_draw(filename, model, ground_data, estimate_data, plot_mode, Ipython, save_img=save_img)

def test_brute_force(filename, test_num, model, isplot=True, plot_mode='3D', Ipython=False, cbr_num=50):
	ground_data, estimate_data = simulation_ukf_brute_force(filename, test_num, model, cbr_num)
	if isplot:
		skeleton_draw(ground_data, estimate_data, test_num=10)

def test_all_save_img(filename, test_num, Ipython=False, cbr_num=50):
	test(filename, test_num, 'IJU', True, 'point', Ipython, cbr_num, save_img=True)
	test(filename, test_num, 'IJU', True, 'length', Ipython, cbr_num, save_img=True)
	test(filename, test_num, 'MJU', True, 'point', Ipython, cbr_num, save_img=True)
	test(filename, test_num, 'MJU', True, 'length', Ipython, cbr_num, save_img=True)

if __name__ == '__main__':
	test_all_save_img("data/input_stand.txt", 50, False, 50)
	# test("data/input_stand.txt", 50, 'MJU', False)
	# test("data/input_stand.txt", 50, 'IJU', False)
	# test("data/input_stand.txt", 50, 'IJU', True, 'point', False, 50, save_img=True)
	# test("data/input_stand.txt", 50, 'MJU', True, 'point', False, 50, save_img=True)
	# test("data/input_stand.txt", 50, 'IJU', True)
	# test_brute_force("data/input_stand.txt", 50, 'IJU', True)
