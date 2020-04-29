import sys
sys.path.append('./src/')
from main import *

def test(filename, test_num, model, isplot=True, plot_mode='3D', Ipython=False, cbr_num=50):
	ground_data, estimate_data = simulation_ukf(filename, test_num, model, cbr_num)
	if isplot:
		skeleton_draw(ground_data, estimate_data, plot_mode, Ipython)

def test_brute_force(filename, test_num, model, isplot=True, plot_mode='3D', Ipython=False, cbr_num=50):
	ground_data, estimate_data = simulation_ukf_brute_force(filename, test_num, model, cbr_num)
	if isplot:
		skeleton_draw(ground_data, estimate_data, test_num=10)

if __name__ == '__main__':
	test("data/input_sitting.txt", 50, 'MJU', True, '3D', False, 50)
	# test("data/input_stand.txt", 50, 'IJU', True)
	# test_brute_force("data/input_stand.txt", 50, 'IJU', True)
