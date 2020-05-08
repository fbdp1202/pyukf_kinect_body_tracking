import sys
import argparse
sys.path.append('./src/')
from main import *

def test_skeleton_filter(person_name, pose_mode, test_num, cbr_num, model):
	filename = merge_skeleton_data('data/skeleton_data/' + person_name + '/' + pose_mode)
	ground_data, estimate_data = simulation_ukf(filename, test_num, cbr_num, model)
	save_skeleton_data_to_csv(person_name, pose_mode, ground_data, estimate_data, model)

def test_skeleton_draw(person_name, pose_mode, model):
	ground_data, estimate_data = read_skeleton_data_from_csv(person_name, pose_mode, model)
	skeleton_draw(person_name, pose_mode, model, ground_data, estimate_data)

def test_skeleton(person_name, pose_mode, test_num, cbr_num, onFilter, model, onPlot):
	if onFilter == 'on':
		test_skeleton_filter(person_name, pose_mode, test_num, cbr_num, model)
	if onPlot == 'on':
		test_skeleton_draw(person_name, pose_mode, model)

def test_one_person_all_mode(person_name, test_num, cbr_num, onFilter, model, onPlot):
	folder_name = 'data/skeleton_data/' + person_name
	dir_list = get_dir_name(folder_name)
	for pose_mode in dir_list:
		print(person_name, pose_mode)
		test_skeleton(person_name, pose_mode, test_num, cbr_num, onFilter, model, onPlot)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('person_name', type=str,
			metavar='First_string',
			help='What is your person name?')
	parser.add_argument('pose_mode', type=str,
			metavar='Second_string',
			help='What is your pose? "*" is all pose')

	parser.add_argument('--filter', type=str, default='on',
			choices=['on','off'],
			help='Do you want to plot?')
	parser.add_argument('--model', type=str, default='ukf',
			choices=['ukf','kf'],
			help='which type of filter?')
	parser.add_argument('--plot', type=str, default='on',
			choices=['on','off'],
			help='Do you want to plot?')
	parser.add_argument('--num', type=int, default=1e9,
			help='How many tests do you want?')
	parser.add_argument('--cbr_num', type=int, default=1e9,
			help='How many calibration tests do you want?')

	args = parser.parse_args()
	
	person_name = args.person_name
	pose = args.pose_mode
	onFilter = args.filter
	model = args.model
	onPlot = args.plot
	test_num = args.num
	cbr_num = args.cbr_num

	if pose == '*':
		test_one_person_all_mode(person_name, test_num, cbr_num, onFilter, model, onPlot)
	else:
		test_skeleton(person_name, pose, test_num, cbr_num, onFilter, model, onPlot)

	# test_skeleton_filter("jiwon", "crossing_arms_30sec", test_num=10)
	# test_skeleton_draw("jiwon", "crossing_arms_30sec", plot_3D=True)
