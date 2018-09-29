# -*- coding: UTF-8 -*-

# have some problem
# there is not a video index
# what can we do add it
import cv2
import sys
import numpy as np
import os

from PIL import Image


def mkdir(path):
    # 引入模块
    # import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return False


# 保存图片
def save_change(save_dir, pil_im, n, box,dir_name):
    box = (box[0],box[1],box[0]+box[2],box[1]+box[3])

    region = pil_im.crop(box)

    out = region.resize((128, 128))
    out_RGB = out.convert('RGB')
    save_dir = save_dir + "_" +dir_name + '/'+"image_tmp_"+ '%05d' % int(n) + ".jpg"
    print(save_dir)
    out_RGB.save(save_dir)


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')



mkdir("person_jpg")
mkdir("person_jpg/waiting")
mkdir("person_jpg/setting")
mkdir("person_jpg/digging")
mkdir("person_jpg/falling")
mkdir("person_jpg/spiking")
mkdir("person_jpg/blocking")
mkdir("person_jpg/jumping")
mkdir("person_jpg/moving")
mkdir("person_jpg/standing")

PATH = "volley_data_big" # source path
all_param = []
for dir_name in os.listdir(PATH):
    with open(os.path.join(PATH, dir_name) + "/annotations.txt") as f:
        lines = f.readlines()
        for line in lines:
            param = line.strip().split(" ")
            all_param.append(param)

        print(param)
    all_path_dir = os.listdir(os.path.join(PATH, dir_name))
    listdir_path = [path for path in all_path_dir if os.path.isdir(os.path.join(PATH, dir_name,path))]
    # listdir_path = [path for path in all_path_dir if path != "annotations.txt"]
    for dirsub_name in listdir_path:

        directory_path = os.path.join(PATH, dir_name, dirsub_name)
        init_frame_path = os.path.join(directory_path, dirsub_name) + ".jpg"
        frame = Image.open(init_frame_path)
        img = np.asarray(frame)
        # Define an initial bounding box
        data = [data for data in all_param if data[0] == (dirsub_name + ".jpg")]
        num_perpson = int((len(data[0]) - 2) / 5)
        for person in range(num_perpson):
            # define tracker

            tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
            tracker_type = tracker_types[2]

            if int(minor_ver) < 3:
                tracker = cv2.Tracker_create(tracker_type)
            else:
                if tracker_type == 'BOOSTING':
                    tracker = cv2.TrackerBoosting_create()
                if tracker_type == 'MIL':
                    tracker = cv2.TrackerMIL_create()
                if tracker_type == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                    tracker2 = cv2.TrackerKCF_create()
                if tracker_type == 'TLD':
                    tracker = cv2.TrackerTLD_create()
                if tracker_type == 'MEDIANFLOW':
                    tracker = cv2.TrackerMedianFlow_create()
                if tracker_type == 'GOTURN':
                    tracker = cv2.TrackerGOTURN_create()
            # aa = all_param.index(dirsub_name + ".jpg")
            bbox_person = data[0][person * 5 + 2: person * 5 + 7]
            bbox_person_int = [int(num) for num in bbox_person[:4]]
            bbox_person_tup = tuple(bbox_person_int)
            # bbox_person = (287, 23, 86, 320)
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(img, bbox_person_tup)

            num_frame = int(dirsub_name)
            for i in range(20):
                current_frame = int(dirsub_name) + i + 1
                current_path = os.path.join(directory_path, str(current_frame)) + ".jpg"
                frame = Image.open(current_path)
                img_current = np.asarray(frame)

                ok, bbox_person_current = tracker.update(img_current)
                if ok:
                    action_class = bbox_person[-1]
                    person_action_class_num_frame = os.path.join("person_jpg", action_class, str(num_frame)+ "-"+str(person))
                    mkdir(person_action_class_num_frame + "_"+dir_name)
                    save_change(person_action_class_num_frame, frame, str(i+21) , bbox_person_current , dir_name)

            bbox_person_tup = tuple(bbox_person_int)
            # bbox_person = (287, 23, 86, 320)
            # Initialize tracker with first frame and bounding box
            ok = tracker2.init(img, bbox_person_tup)


            for i in range(20):
                current_frame = int(dirsub_name) - i
                current_path = os.path.join(directory_path, str(current_frame)) + ".jpg"

                frame = Image.open(current_path)
                img_current = np.asarray(frame)

                ok, bbox_person_current = tracker2.update(img_current)
                if ok:
                    action_class = bbox_person[-1]
                    person_action_class_num_frame = os.path.join("person_jpg", action_class, str(num_frame)+ "-"+str(person))
                    mkdir(person_action_class_num_frame + "_"+dir_name)
                    save_change(person_action_class_num_frame, frame, str(20-i-1) , bbox_person_current,dir_name)
            # save itself
            frame = Image.open(init_frame_path)
            person_action_class_num_frame = os.path.join("person_jpg", action_class, str(num_frame) + "-" + str(person))
            save_change(person_action_class_num_frame, frame, str(20), bbox_person_tup,dir_name)

"""
下面代码在转视频的时候可以用到！！

with open("volley_data/39/annotations.txt") as f:

    lines = f.readlines()
    for line in lines :
        param = line.strip().split(" ")
    print(param)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    # Read video
    video = cv2.VideoCapture("00001.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

    frame = Image.open("39/29885/29885.jpg")
    img = np.asarray(frame)
    bbox = cv2.selectROI(img, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(img, bbox)

    for i in range(5):
        n = 29885 - i - 1
        frame = Image.open("39/29885/29886.jpg")

    for i in range(4):
        n = 29885 + i + 1
        frame = Image.open("39/29885/29886.jpg")
        img = np.asarray(frame)
        # Update tracker
        ok, bbox = tracker.update(img)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
            # Display result
            cv2.imshow("Tracking", img)
            """

