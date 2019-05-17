import os
import matplotlib
import cv2
import re

def load_data(file_name):
    train_dict = dict()
    train_imgs = []  # useful to keep an ordering of the imgs
    with open(file_name, "r") as file:
        for line in file:
            info = re.split(",| ", line[:-1])  # the last character is '\n'
            img_path = info[0]
            train_imgs.append(img_path)
            bounding_boxes = []  # with the label
            for i in range(1, len(info), 5):
                bounding_boxes.append([int(value) for value in info[i:i + 5]])

            train_dict[img_path] = bounding_boxes

    return train_dict

def main():
    f_o = "lisa_c256.txt"
    d_o = load_data(f_o)
    f_n = "cgan2_lisa_c256.txt"
    d_n = load_data(f_n)

    d_o_lookup = {os.path.splitext(os.path.basename(key))[0] : key for key in d_o}

    dir_new = os.path.splitext(f_n)[0] + "_inserted"

    if not os.path.exists(dir_new):
        os.makedirs(dir_new)

    for path_n, bboxes in d_n.items():
        name_n = os.path.splitext(os.path.basename(path_n))[0]
        path_o = d_o_lookup[name_n]
        img_o = cv2.imread(path_o)
        img_n = cv2.imread(path_n)

        for bb in bboxes:
            x0 = int(bb[0])
            y0 = int(bb[1])
            x1 = int(bb[2])
            y1 = int(bb[3])
            img_n[y0:y1,x0:x1] = img_o[y0:y1,x0:x1]
        
        path_list = os.path.normpath(path_n).split(os.sep)
        path_list[0] = dir_new
        new_path = os.path.join(*path_list)

        new_dir = os.path.dirname(new_path)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            
        print(new_path)
        cv2.imwrite(new_path, img_n)
            

if __name__ == '__main__':
    main()