# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Tusimple
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Songyang Zhang
# --------------------------------------------------------
import os
kitti_mot_image_path_train = '/data/syzhang/kitti_mot/training_original/image_02/'
kitti_mot_label_path_train = '/data/syzhang/kitti_mot/training_original/label_02/'

#kitti_mot_path_test = '/data/syzhang/kitti_mot/training_test/label_02/'

def create_index_file_train(kitti_mot_path):
    label_file_path = []
    for root, dirs, files in os.walk(kitti_mot_path):
        for item in files:
            item_path = os.path.join(root, item)
            label_file_path.append(item_path)
    label_file_path.sort()

    path_to_save = './kitti_mot_train_new.txt'
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

# def create_annotaion_file_train(kitti_mot_path):
#     classes =  ['Car', 'Pedestrian', 'Cyclist']
#     label_file_path = []
#     for root, dirs, files in os.walk(kitti_mot_path):
#         for item in files:
#             item_path = os.path.join(root, item)
#             label_file_path.append(item_path)
#     label_file_path.sort()
#
#     # path_to_save = './kitti_mot_val.txt'
#     # if os.path.exists(path_to_save):
#     #     os.remove(path_to_save)
#
#
#     for item in label_file_path:
#         with open(item, 'r') as fid:
#             annotations_list = fid.readlines()
#
#             frame_id_list = [ ]
#             video_ID = item.strip().split('/')[-1].replace('.txt', '')
#
#             # annotation save folder
#             annotation_save_dir = kitti_mot_path.replace('label_02', 'label_new_2')
#             annotation_save_dir = os.path.join(annotation_save_dir, video_ID)
#
#             if os.path.exists(annotation_save_dir):
#                 import shutil
#                 shutil.rmtree(annotation_save_dir)
#                 os.makedirs(annotation_save_dir)
#             else:
#                 os.makedirs(annotation_save_dir)
#
#
#             for annotation in annotations_list:
#                 annotation_ = annotation.strip().split(' ')
#                 frame_id = int(annotation_[0])
#                 annotation_file_path = os.path.join(annotation_save_dir, str(frame_id).zfill(6)+'.txt')
#                 category = annotation_[2]
#                 if category in classes:
#                     with open(annotation_file_path, 'a') as fid:
#                         fid.write(annotation)
#
#             #     frame_id_list.append(frame_id)
#             #
#             #
#             # for frame_seg_id in frame_id_list:
#             #     image_path = item.replace('label_02', 'image_02')
#             #     image_path = image_path.replace('.txt', '')
#             #     assert os.path.exists(image_path), "{} file not exists".format(image_path)
#             #
#             #     frame_id = 1
#             #     flist.write("{:s} {:d} {:d} {:d}\n".format(image_path, frame_id, frame_seg_id, frame_len))

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

            annotations_dict = dict()

            for annotations in annotations_list:
                annotations_ = annotations.strip().split(' ')
                frame_id = int(annotations_[0])

                frame_id_list.append(frame_id)

            frame_max = max(frame_id_list)

            for annotation in annotations_list:
                annotation_ = annotation.strip().split(' ')
                frame_id = int(annotation_[0])
                annotation_file_path = os.path.join(annotation_save_dir, str(frame_id).zfill(6)+'.txt')
                category = annotation_[2]
                if category in classes:
                    with open(annotation_file_path, 'a') as fid:
                        fid.write(annotation)
            for i in range(frame_max+1):
                annotation_file_path = os.path.join(annotation_save_dir, str(i).zfill(6)+'.txt')
                if not os.path.exists(annotation_file_path):
                    with open(annotation_file_path, 'w') as f:
                        # f.write(None)
                        print annotation_file_path

def create_index_file_val(kitti_mot_path):
    classes =  ['Car', 'Pedestrian', 'Cyclist']
    label_file_path = []
    for root, dirs, files in os.walk(kitti_mot_path):
        for item in files:
            item_path = os.path.join(root, item)
            label_file_path.append(item_path)
    label_file_path.sort()

    path_to_save = './data/kitti/kitti_mot_training_test.txt'
    if os.path.exists(path_to_save):
        os.remove(path_to_save)

    total_image_number = 1

    for item in label_file_path:
        with open(item, 'r') as fid:
            annotations_list = fid.readlines()

            frame_id_list = [ ]
            video_ID = item.strip().split('/')[-1].replace('.txt', '')

            # annotation save folder
            annotation_save_dir = kitti_mot_path.replace('label_02', 'image_02')
            annotation_save_dir = os.path.join(annotation_save_dir, video_ID)


            for annotations in annotations_list:
                annotations_ = annotations.strip().split(' ')
                frame_id = int(annotations_[0])

                frame_id_list.append(frame_id)

            frame_max = max(frame_id_list)

            with open(path_to_save, 'a') as f:
                print path_to_save
                f.write("{} {} {} {}\n".format(annotation_save_dir, total_image_number, 0, frame_max))

            total_image_number += frame_max

def main():
    create_index_file_train(kitti_mot_label_path_train)
    create_annotaion_file(kitti_mot_label_path_train)

if __name__ == '__main__':
    main()
