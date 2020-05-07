import sys
sys.path.append('./src/')
from main import *

def test_skeleton_filter(person_name, pos_mode, test_num=1e9, cbr_num=1e9, model='ukf'):
	filename = merge_skeleton_data('data/skeleton_data/' + person_name + '/' + pos_mode)
	ground_data, estimate_data = simulation_ukf(filename, test_num, cbr_num, model)
	save_skeleton_data_to_csv(person_name, pos_mode, ground_data, estimate_data, model)

def test_skeleton_draw(person_name, pos_mode, plot_3D=False, model='ukf'):
	ground_data, estimate_data = read_skeleton_data_from_csv(person_name, pos_mode, model)
	skeleton_draw(person_name, pos_mode, model, ground_data, estimate_data, plot_3D)

def test_all_save_img(filename, test_num=1e9, cbr_num=1e9):
	test(filename, test_num, 'ukf', True, 'point', cbr_num, save_img=True)
	test(filename, test_num, 'ukf', True, 'length', cbr_num, save_img=True)

def test_one_person_all_mode(person_name, test_num=1e9, cbr_num=1e9, model='ukf'):
	folder_name = 'data/skeleton_data/' + person_name
	dir_list = get_dir_name(folder_name)
	for pos_mode in dir_list:
		test_skeleton_filter(person_name, pos_mode)

if __name__ == '__main__':
	# test_skeleton_filter("jiwon", "crossing_arms_30sec", test_num=10)
	test_skeleton_draw("jiwon", "crossing_arms_30sec", True)
	# test_one_person_all_mode("jiwon")
	# test("data/input_stand.txt", 50, 'ukf', True, 'point', False, 50, save_img=True)
