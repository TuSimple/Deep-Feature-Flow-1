# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Tusimple
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Songyang Zhang
# --------------------------------------------------------
import os
kitti_mot_path = '/data/syzhang/kitti_mot/training/label_02/val'
def create_index_file(kitti_mot_path):
    label_file_path = []
    for root, dirs, files in os.walk(kitti_mot_path):
        for item in files:
            item_path = os.path.join(root, item)
            label_file_path.append(item_path)
    label_file_path.sort()

    path_to_save = './kitti_mot_val.txt'
    if os.path.exists(path_to_save):
        os.remove(path_to_save)

    with open(path_to_save, 'a') as flist:
        for item in label_file_path:
            with open(item, 'r') as fid:
                annotations_list = fid.readlines()

                frame_id_list = [ ]
                for annotation in annotations_list:
                    annotation_ = annotation.strip().split(' ')
                    frame_id = int(annotation_[0])
                    frame_id_list.append(frame_id)
                    category = annotation_[2]
                frame_id_list = list(set(frame_id_list))

                del frame_id_list[0]
                frame_len = max(frame_id_list)
                print frame_len
                for frame_seg_id in frame_id_list:
                    image_path = item.replace('label_02', 'image_02')
                    image_path = image_path.replace('.txt', '')
                    assert os.path.exists(image_path), "{} file not exists".format(image_path)

                    frame_id = 1
                    flist.write("{:s} {:d} {:d} {:d}\n".format(image_path, frame_id, frame_seg_id, frame_len))

def create_annotaion_file(kitti_mot_path):
    classes =  ['Car', 'Pedestrian', 'Cyclist']
    label_file_path = []
    for root, dirs, files in os.walk(kitti_mot_path):
        for item in files:
            item_path = os.path.join(root, item)
            label_file_path.append(item_path)
    label_file_path.sort()

    # path_to_save = './kitti_mot_val.txt'
    # if os.path.exists(path_to_save):
    #     os.remove(path_to_save)


    for item in label_file_path:
        with open(item, 'r') as fid:
            annotations_list = fid.readlines()

            frame_id_list = [ ]
            video_ID = item.strip().split('/')[-1].replace('.txt', '')

            # annotation save folder
            annotation_save_dir = kitti_mot_path.replace('label_02', 'label_new')
            annotation_save_dir = os.path.join(annotation_save_dir, video_ID)

            if os.path.exists(annotation_save_dir):
                import shutil
                shutil.rmtree(annotation_save_dir)
                os.makedirs(annotation_save_dir)
            else:
                os.makedirs(annotation_save_dir)


            for annotation in annotations_list:
                annotation_ = annotation.strip().split(' ')
                frame_id = int(annotation_[0])
                annotation_file_path = os.path.join(annotation_save_dir, str(frame_id).zfill(6)+'.txt')
                category = annotation_[2]
                if category in classes:
                    with open(annotation_file_path, 'a') as fid:
                        fid.write(annotation)

            #     frame_id_list.append(frame_id)
            #
            #
            # for frame_seg_id in frame_id_list:
            #     image_path = item.replace('label_02', 'image_02')
            #     image_path = image_path.replace('.txt', '')
            #     assert os.path.exists(image_path), "{} file not exists".format(image_path)
            #
            #     frame_id = 1
            #     flist.write("{:s} {:d} {:d} {:d}\n".format(image_path, frame_id, frame_seg_id, frame_len))

create_annotaion_file(kitti_mot_path)
