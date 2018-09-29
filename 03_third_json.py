from __future__ import print_function, division
import os
import sys
import json
import pandas as pd
# param
# python /home/ubuntu/ZhangXianJie/action_recongnition/third_json.py person_jpg
def convert_csv_to_dict(csv_dir_path):
    database = {}
    for filename in os.listdir(csv_dir_path):

        class_path = os.path.join(csv_dir_path, filename)
        # for sub_class in os.listdir(class_path):
        #     w = 1

        keys = os.listdir(class_path)
        val_num = len(keys) // 5

        subsets = []
        for i in range(len(keys)):

            if i >= val_num:
                subset = 'training'
            elif i <= val_num:
                subset = 'validation'
            subsets.append(subset)

        for i in range(len(keys)):
            key = keys[i]
            database[key] = {}
            database[key]['subset'] = subsets[i]
            label = filename
            database[key]['annotations'] = {'label': label}
    
    return database

def get_labels(csv_dir_path):
    labels = []
    for name in os.listdir(csv_dir_path):
        labels.append(name)
    return sorted(list(set(labels)))

def convert_csv_to_activitynet_json(csv_dir_path, dst_json_path):
    labels = get_labels(csv_dir_path)
    database = convert_csv_to_dict(csv_dir_path)
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

if __name__ == '__main__':
    csv_dir_path = sys.argv[1]


    dst_json_path = 'volley_data.json'
    convert_csv_to_activitynet_json(csv_dir_path, dst_json_path)