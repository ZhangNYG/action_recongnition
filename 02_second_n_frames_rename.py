from __future__ import print_function, division
import os
import sys
import subprocess


# add param   python second_n_frames.py person_jpg

def class_process(dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    for file_name in os.listdir(class_path):
        video_dir_path = os.path.join(class_path, file_name)
        image_indices = []
        for image_file_name in os.listdir(video_dir_path):
            if 'image' not in image_file_name:
                continue
            index_list = image_file_name.strip().strip(".jpg").split("_")
            num_index = index_list[1]
            if 'tmp'  in image_file_name:
                num_index = index_list[2]
            image_indices.append(int(num_index))

        if len(image_indices) == 0:
            print('no image files', video_dir_path)
            n_frames = 0
        else:
            image_indices.sort(reverse=True)
            n_frames = int(image_indices[0]) - int(image_indices[-1]) + 1
            print(video_dir_path, n_frames)
        with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
            dst_file.write(str(n_frames))
        image_indices.sort()
        for i in range(len(image_indices)):

            src_path = video_dir_path + "/image_tmp_" + "%05d" % int(image_indices[i]) + ".jpg"
            if os.path.isfile(src_path):
                dst_path = video_dir_path + "/image_" + "%05d" % (i + 1) + ".jpg"
                os.rename(src_path,  dst_path)


if __name__ == "__main__":
    dir_path = sys.argv[1]
    for class_name in os.listdir(dir_path):
        class_process(dir_path, class_name)
