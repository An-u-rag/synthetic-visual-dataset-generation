import numpy as np
import os
import json
import os

# class_name_to_id_mapping = {"Cow": 0,
#                 "Chicken": 1,
#                 "Sheep": 2,
#                 "Goat": 3,
#                 "Pig": 4}
class_name_to_id_mapping = {"cow_1": 0,
                            "cow_2": 1,
                            "cow_3": 2,
                            "cow_4": 3,
                            "cow_5": 4,
                            "pig_clean": 5,
                            "pig_dirty": 6
                            }

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict, image_file, name_file):
    print_buffer = []
    print(info_dict)
    data = np.load(info_dict)
    # image_file = Image.open(image_file)
    image_w, image_h = image_file.size
    class_id = {}
    with open(name_file, 'r') as info_name:
        data_name = json.load(info_name)
        # print(data_name)
        for k, v in data_name.items():
            class_id[k] = class_name_to_id_mapping[v["class"]]
        # for values in data_name.values():
        #     # print(values)
        #     class_id[values["class"]] = class_name_to_id_mapping[values["class"]]
        #     class_id[class_name_to_id_mapping[values["class"]]] = values["class"]
        #     class_id.append(class_name_to_id_mapping[values["class"]])
            # class_id = class_name_to_id_mapping[values["name"]]
        # print(class_id)
    # counter = 0
    # For each bounding box
    for b in data:
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b[1] + b[3]) / 2
        b_center_y = (b[2] + b[4]) / 2
        b_width = (b[3] - b[1])
        b_height = (b[4] - b[2])

        # Normalise the co-ordinates by the dimensions of the image
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # print(counter)
        print(class_id)
        # Write the bbox details to the file
        print(class_id.get(str(b[0])))
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id.get(str(b[0])), b_center_x, b_center_y, b_width, b_height))
        # counter += 1
        # print(print_buffer)
    # Name of the file which we have to save
    path_pic = "C:/Users/xyche/Downloads/dataset"
    save_file_name = os.path.join(path_pic, info_dict.replace("bounding_box_2d_tight_", "rgb_").replace("npy", "txt"))

    # Save the annotation to disk
    print("\n".join(print_buffer), file=open(save_file_name, "w"))


import os

from PIL import Image

# Convert and save the annotations
# path_label = "/content/RenderProduct_Replicator/bounding_box_2d_loose"
path_pic = "C:/Users/xyche/Downloads/dataset"
datanames = os.listdir(path_pic)
for i in datanames:
    if os.path.splitext(i)[1] == '.npy':
        # np.load("../"+info_dict)

        # info_dict = open(os.path.join(path_pic,i), "rb")
        info_dict = os.path.join(path_pic, i)

        image_file = i.replace("bounding_box_2d_tight_", "rgb_").replace("npy", "png")

        # os.listdir(path_pic)
        image_file = Image.open(os.path.join(path_pic, image_file))

        info_name = i.replace("bounding_box_2d_tight_", "bounding_box_2d_tight_labels_").replace("npy", "json")
        name_file = os.path.join(path_pic, info_name)

        convert_to_yolov5(info_dict, image_file, name_file)
# print(os.listdir(path_pic))
annotations = [os.path.join(path_pic, x) for x in os.listdir(path_pic) if x[-3:] == "txt" and x != 'metadata.txt']
# print(len(annotations))

from sklearn.model_selection import train_test_split

# Read images and annotations
images = [os.path.join(path_pic, x) for x in os.listdir(path_pic) if x[-3:] == "png"]
# print(len(images))
# datanames = os.listdir(path_pic)
annotations = [os.path.join(path_pic, x) for x in os.listdir(path_pic) if x[-3:] == "txt" and x != 'metadata.txt']
# print(len(annotations))

images.sort()
annotations.sort()
# for i in annotations:
#     update_annotations = i.replace("bounding_box_2d_loose_", "rgb_").replace("txt", "png")
#     if update_annotations not in images:
#         print(update_annotations)

# Split the dataset into train-valid-test splits
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.2,
                                                                                random_state=1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,
                                                                              test_size=0.5, random_state=1)

path1 = 'C:/Users/xyche/Downloads/dataset'

os.mkdir(path1 + '/images')
os.mkdir(path1 + '/labels')
file_name = ['/train', '/val', '/test']
path2 = 'C:/Users/xyche/Downloads/dataset/images'
for name in file_name:
    os.mkdir(path2 + name)
path3 = 'C:/Users/xyche/Downloads/dataset/labels'
for name in file_name:
    os.mkdir(path3 + name)

import shutil


# Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        shutil.copy(f, destination_folder)


# Move the splits into their folders
move_files_to_folder(train_images, 'C:/Users/xyche/Downloads/dataset/images/train/')
move_files_to_folder(val_images, 'C:/Users/xyche/Downloads/dataset/images/val/')
move_files_to_folder(test_images, 'C:/Users/xyche/Downloads/dataset/images/test/')
move_files_to_folder(train_annotations, 'C:/Users/xyche/Downloads/dataset/labels/train/')
move_files_to_folder(val_annotations, 'C:/Users/xyche/Downloads/dataset/labels/val/')
move_files_to_folder(test_annotations, 'C:/Users/xyche/Downloads/dataset/labels/test/')

import yaml
desired_caps = {
                'train': 'C:/Users/xyche/Downloads/dataset/images/train/',
                'val':  'C:/Users/xyche/Downloads/dataset/images/val/',
                'test': 'C:/Users/xyche/Downloads/dataset/images/test/',

                # number of classes
                'nc': 7,

                # class names
                #'names': ['Sam', 'Lucy', 'Ross', 'Mary', 'Elon', 'Alex', 'Max']
                'names': ['0', '1', '2', '3', '4', '5', '6']
                }

curpath = 'C:/Users/xyche/Downloads/dataset'
yamlpath = os.path.join(curpath, "./dataset.yaml")

with open(yamlpath, "w", encoding="utf-8") as f:
    yaml.dump(desired_caps, f)