import os
import json
import numpy as np
from tqdm import tqdm

ANNOTATION_ROOT = "cityscape/gtFine"
IMAGE_ROOT = "cityscape/leftImg8bit"


def same_picture(json_path, png_path):
    """
    Extract and compare names from the file paths.
    example:
    same_picture("a/b/c/d/name1_name2_name3_xxxx.json,
                 "e/f/g/h/name1_name2_name3_yyyy.png) == True
    same_picture("a/b/c/d/name1_othername_name3_xxxx.json,
                 "e/f/g/h/name1_name2_name3_yyyy.png) == False
    """
    json_name = "_".join(json_path.split("/")[4].split("_")[0:3])
    png_name = "_".join(png_path.split("/")[4].split("_")[0:3])
    return png_name == json_name


def get_filepaths(root_dir, file_type):
    """
    get all the .json or .png file paths for a folder with this structure:
    root_dir
        - dir1
            - file1.json (or .png)
            - file2.json (or .png
            - file3.json (or .png)
        - dir2
            - file4.json
            - file5.json
            - undesired_file.other_ext
        - dir3
        ...
    :param root_dir: str
    :param file_type: str: either "json" or "png"
    :return:
    """
    assert(file_type == "json" or file_type == "png")
    filepaths = []
    walk = os.walk(root_dir)
    directories = next(walk)[1]
    for directory in directories:
        files = next(walk)[2]
        for file in files:
            if '.' + file_type in file:
                filepaths.append(root_dir + '/' + directory + '/' + file)
    return filepaths


def pol_2_bb(polygons):
    """
    Given polygones, return the corresponding bounding boxes in the following format:
    x_min, y_min, x_max, y_max
    """
    bbs = []
    for polygon in polygons:
        polygon = np.array(polygon)
        x_min = np.min(polygon[:, 0])
        y_min = np.min(polygon[:, 1])
        x_max = np.max(polygon[:, 0])
        y_max = np.max(polygon[:, 1])
        bbs.append((x_min, y_min, x_max, y_max))
    return bbs


def get_signs_polygons(info_dict):
    """
    Given the info_dict of an image, returns the polygons of the signs.
    """
    polygons = []
    for obj in info_dict["objects"]:
        if obj["label"] == "traffic sign":
            polygons.append(obj["polygon"])
    return polygons


def get_signs_bbs(info_dict):
    polygons = get_signs_polygons(info_dict)
    bbs = pol_2_bb(polygons)
    return bbs


def write_input_file(json_root, png_root, file_name):
    json_filepaths = get_filepaths(json_root, "json")
    png_filepaths = get_filepaths(png_root, "png")

    with open(file_name, "w") as input_file:
        for json_path, png_path in tqdm(zip(json_filepaths, png_filepaths),
                                        total=len(json_filepaths),
                                        ncols=100,
                                        ascii=True):
            assert (same_picture(json_path, png_path))  # making sure we have the good (annotation, image) pair
            with open(json_path, "r") as file:
                info_dict = json.load(file)
            bounding_boxes = get_signs_bbs(info_dict)

            input_file.write(png_path)
            for bb in bounding_boxes:
                input_file.write(" {},{},{},{},0"
                                 .format(bb[0], bb[1], bb[2], bb[3]))
            input_file.write("\n")


if __name__ == "__main__":
    for data_split in ["train", "test"]:
        print("Pre-processing CITYSCAPE {} dataset...".format(data_split))
        json_root = ANNOTATION_ROOT + "/" + data_split
        png_root = IMAGE_ROOT + "/" + data_split
        file_name = "{}_cityscape.txt".format(data_split)

        write_input_file(json_root, png_root, file_name)
        print("Done.")
        print()