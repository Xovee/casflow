# Author: Xovee Xu
# Email: xovee at live.com

dataset = 'xovee'  # modify this to your own dataset directory

# you should use different time settings for different datasets
observation_time = 3600*.5
prediction_time = [24*3600]

max_sequence = 100
gc_emd_size = 40
gg_emd_size = 40
number_of_s = 2
number_of_flows = 8

seed = 'xovee'

# dataset path
data_path = "dataset/" + dataset

cascade_path = data_path + "/dataset.txt"
cascade_filtered_path = data_path + '/dataset_filtered.txt'
cascade_sorted_filtered_path = data_path + '/dataset_sorted_filtered.txt'

cascade_train = data_path + "/cascade_train.txt"
cascade_validation = data_path + "/cascade_validation.txt"
cascade_test = data_path + "/cascade_test.txt"

cascade_shortestpath_train = data_path + "/cascade_shortestpath_train.txt"
cascade_shortestpath_validation = data_path + "/cascade_shortestpath_validation.txt"
cascade_shortestpath_test = data_path + "/cascade_shortestpath_test.txt"

global_graph = data_path + "/global_graph.pkl"
global_embedding = data_path + "/gg"

train = data_path + '/data_train.pkl'
val = data_path + '/data_val.pkl'
test = data_path + '/data_test.pkl'
