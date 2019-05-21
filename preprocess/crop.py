import re
import cv2
import itertools
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Bounding box cropping script")
parser.add_argument("file_name", type=str, help="The .txt annotations file")
parser.add_argument("crop_size", type=int, help="The size of the square crops")

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
    args = parser.parse_args()

    if args.file_name == "":
        parser.error("Specify a file_name")

    if args.crop_size <= 0:
        parser.error("Specify a crop_size > 0")

    file_name = args.file_name
    crop_size = args.crop_size
    np.random.seed(42)

    data_dict = load_data(file_name)
    new_file_name = os.path.splitext(file_name)[0] + "_crop_%i" % crop_size + ".txt"

    with open(new_file_name, "w") as f:
        for path, bboxes in data_dict.items():
            img = cv2.imread(path)
            crop, bb = get_crop(img, crop_size, bboxes)
            if crop is not None:
                crop_img = img[crop[1]:crop[3], crop[0]:crop[2]]
                path_list = os.path.normpath(path).split(os.sep)
                path_list[0] += "_crop_%i" % crop_size
                new_path = os.path.join(*path_list)
                new_dir = os.path.dirname(new_path)
                print(new_path)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                cv2.imwrite(new_path, crop_img)

                f.write(new_path)
                for b in bb:
                    f.write(" %s,%s,%s,%s,%s" % (b[0] - crop[0], b[1] - crop[1], b[2] - crop[0], b[3] - crop[1], b[4]))
                f.write("\n")
        
        #print("Image: %s, Crop: %s, cropped bboxes: %s, Original bboxes: %s" % (path, crop, bb, bboxes))

def get_crop(img, crop_size, bboxes):
    h, w, _ = img.shape

    options = []

    for l in range(0, len(bboxes) + 1):
        for combination in itertools.combinations(bboxes, l):
            options.append(combination)
        
    options.sort(key=lambda o: len(o), reverse=True)
    # Itertools produces empty elements? Remove them:
    options = list(filter(None, options))
    
    for option in options:
        crop = find_best_crop(w, h, crop_size, option)
        if crop is not None:
            return crop, option

    return None, None
    
def find_best_crop(w, h, crop_size, bboxes):
    # Find bounds
    bx0 = min(box[0] for box in bboxes)
    by0 = min(box[1] for box in bboxes)
    bx1 = max(box[2] for box in bboxes)
    by1 = max(box[3] for box in bboxes)
    # Find width and height of bounds
    bw = bx1 - bx0
    bh = by1 - by0
    # Check if the bounds will fit within the crop
    if bw <= crop_size and bh <= crop_size:
        # How much can the crop move while still containing the bounds?
        dx = (crop_size - bw) / 2
        dy = (crop_size - bh) / 2
        jitter_x = np.random.randint(-dx, dx + 1)
        jitter_y = np.random.randint(-dy, dy + 1)
        # Find center of the bounds
        cx = bx0 + bw / 2 + jitter_x
        cy = by0 + bh / 2 + jitter_y
        # Find the new bounds of the crop
        cx0 = int(cx - crop_size / 2)
        cy0 = int(cy - crop_size / 2)
        cx1 = int(cx + crop_size / 2)
        cy1 = int(cy + crop_size / 2)
        # Create adjustments in x and y
        px = 0
        py = 0
        # Check if the bounds are outside the image in the x-axis
        if cx0 < 0:
            px = -cx0
        elif cx1 > w:
            px = w - cx1
        # Check if the bounds are outside the image in the y-axis
        if cy0 < 0:
            py = -cy0
        elif cy1 > h:
            py = h - cy1
        # Return bounds of the crop and nudge it so that it is inside the image if needed
        return [cx0 + px, cy0 + py, cx1 + px, cy1 + py]
    else:
        # If the crop wont fit the signs, return None
        return None

if __name__ == "__main__":
    main()