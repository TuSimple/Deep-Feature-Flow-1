
# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Tusimple
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Songyang Zhang
# --------------------------------------------------------
import os
import copy
import numpy as np
def create_index_file_kitti_det():

    RAND_FILE =  './data/kitti/devkit_object/mapping/train_rand.txt'
    MAP_FILE = './data/kitti/devkit_object/mapping/train_mapping.txt'

    DATA_INDEX_FILE = './data/kitti/train.txt'

    rand_map = []
    with open(RAND_FILE) as f:
        for line in f:
            rand_map = line.strip().split(',')

    img_map = []
    with open(MAP_FILE) as f:
        for line in f:
            line_list = line.strip().split(' ')
            img_map.append(line_list)

    train_image_list = []
    with open(DATA_INDEX_FILE) as f:
        for line in f:
            name = os.path.join('./data/kitti/images', line)
            train_image_list.append(name)

    prev_not_exist = []
    with open('./data/kitti/kitti_det_train.txt', 'w') as f:
        for item in train_image_list:
            curr_img_abs_path, prev_img_abs_path = get_pair_path(item, rand_map, img_map)
            # print item
            # print curr_img_abs_path
            # print prev_img_abs_path
            frame_id = 1
            frame_seg_id = int(os.path.basename(curr_img_abs_path).replace('.png', ''))
            frame_seg_len = frame_seg_id
            kitti_det_index = int(os.path.basename(item))
            f.write("{:s} {:d} {:d} {:d} {:d}\n".format(os.path.dirname(curr_img_abs_path), frame_id, frame_seg_id, frame_seg_len, kitti_det_index))

            if not os.path.exists(prev_img_abs_path):
                prev_not_exist.append(item)

        print prev_not_exist
def get_pair_path(curr_img_path, rand_map, img_map):
    RAWDATA_PATH = '/data/syzhang/kitti_rawdata'
    img_name = os.path.basename(curr_img_path)
    img_num = int(os.path.splitext(img_name)[0])
    rand_idx = int(rand_map[img_num])
    curr_img_info = img_map[rand_idx - 1]

    curr = int(curr_img_info[2])

    prev_img_info = copy.deepcopy(curr_img_info)
    prev = curr - 1
    prev_img_info[2] = str(prev).zfill(10)

    curr_img_abs_path = os.path.join(RAWDATA_PATH, curr_img_info[0], curr_img_info[1], 'image_02', 'data', curr_img_info[2] + '.png')

    prev_img_abs_path = os.path.join(RAWDATA_PATH, prev_img_info[0], prev_img_info[1], 'image_02', 'data', prev_img_info[2] + '.png')

    return curr_img_abs_path, prev_img_abs_path
# create_index_file_kitti_det()

def create_annotaion_file_kitti_det():
    path = './data/kitti/kitti_det_total_label.lst'
    new_label_base_path = './data/kitti/kitti_det_label/total/'
    classes = ['car', 'pedestrian', 'cyclist']

    with open(path, 'r') as f:
        lines = [x.strip().split(':') for x in f.readlines()]

    if os.path.exists(new_label_base_path):
        import shutil
        shutil.rmtree(new_label_base_path)
        os.makedirs(new_label_base_path)
    else:
        os.makedirs(new_label_base_path)
    for item in lines:
        frame_index = item[0]
        bbox = item[1:]
        label_file = frame_index.replace('.png','.txt')
        label_file = os.path.join('./data/kitti/kitti_det_label/total/', label_file)

        with open(label_file, 'w') as f:
            for i in range(len(classes)):
                if len(bbox[i]) == 0:
                    continue
                else:
                    class_name = classes[i]
                    class_i_box = map(float, bbox[i].strip().split(' '))
                    boxes_list = np.array(class_i_box, dtype=np.float32).reshape(-1,4).tolist()
                    for box in boxes_list:
                        f.write("{} {} {} {} {}\n".format(class_name, box[0], box[1], box[2], box[3]))

create_annotaion_file_kitti_det()
