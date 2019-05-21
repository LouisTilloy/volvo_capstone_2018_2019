import os
import re
import cv2
import shutil
import argparse
import subprocess

AUG_BLEND = "BLEND"
AUG_SAUG = "SAUG"
AUG_CG = "CG"
AUG_CG_INS = "CG_INS"
AUG_BBG = "BBG"
AUG_AAUG = "AAUG"
AUG_AAUG_BBG = "AAUG_BBG"

parser = argparse.ArgumentParser(description="Augment data set.")
parser.add_argument("method", metavar="method", type=str, help="The augmentation method.")
parser.add_argument("path_ann", metavar="annotations.txt", type=str, help="The path to the txt-file containing the annotations.")
parser.add_argument("dir_out", metavar="dir_out", type=str, help="The path to the directory where the augmented data will be saved.")
args = parser.parse_args()

#if not os.path.isfile(args.path_ann):
#    print("Error: The given annotation file %s does not exist." % (args.path_ann))
#    exit()

#if not os.path.isdir(args.dir_out):
#    print("Error: The given directory %s does not exist." % (args.dir_out))
#    exit()

def load_data(file_name):
    dct = dict()
    with open(file_name, "r") as file:
        for line in file:
            info = re.split(",| ", line.rstrip())
            pth_img = info[0]
            bbs = []  # with the label
            for i in range(1, len(info), 5):
                bbs.append([int(value) for value in info[i:i + 5]])
            dct[pth_img] = bbs
    return dct

def save_data(file_name, data):
    with open(file_name, "w") as f:
        for img, bbs in data.items():
            f.write(img)
            for b in bbs:
                f.write(" %s,%s,%s,%s,%s\n" % (b[0], b[1], b[2], b[3], b[4]))
                    
# Produces a list of supported filetypes from the specified directory.
def get_paths(directory):
    SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]
    paths = []
    for dirpath, _, files in os.walk(directory):
        for f in files:
            if any(f.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                paths.append(os.path.join(dirpath, f))
    return paths
    
def augment(method, path_ann, dir_out):
    path = os.getcwd()
    path_ann = os.path.join(os.getcwd(), path_ann)
    dir_out = os.path.join(os.getcwd(), dir_out)

    if (method == AUG_BLEND):
        aug_blend(path, path_ann, dir_out)
    elif (method == AUG_SAUG):
        aug_saug(path, path_ann, dir_out)
    elif (method == AUG_CG):
        aug_cg(path, path_ann, dir_out)
    elif (method == AUG_CG_INS):
        aug_cg(path, path_ann, dir_out, insert=True)
    elif (method == AUG_BBG):
        pass
    elif (method == AUG_AAUG):
        pass
    elif (method == AUG_AAUG_BBG):
        pass

def aug_blend(path, path_ann, dir_out):
    path_blend = os.path.join(path, "augment/blend/augment.blend")
    path_script = os.path.join(path, "augment/blend/scripts/main.py")
    subprocess.Popen(["blender", path_blend, "--background", "--python", path_script, "--", path_ann, dir_out], shell=True)

def aug_saug(path, path_ann, dir_out):
    path_script = os.path.join(path, "augment/saug/saug.py")
    subprocess.Popen(["python", path_script, path_ann, dir_out], shell=True)

def aug_cg(path, path_ann, dir_out, insert=False):
    path_script = os.path.join(path, "augment/cg/test.py")
    dir_script = os.path.dirname(path_script)
    
    dir_in = os.path.dirname(path_ann)
    dir_temp = os.path.join(dir_in, "temp_cg")
    os.makedirs(dir_temp, exist_ok=True)

    src = load_data(path_ann)
    for img in src.keys():
        name  = os.path.basename(img)
        s = os.path.join(dir_in, img)
        t = os.path.join(dir_temp, name)
        shutil.copy(s, t)

    suffix = "_cg"
    if insert:
        suffix += "_ins"

    paths = get_paths(dir_temp)
    n = str(len(paths))
    ann_name = os.path.splitext(os.path.basename(path_ann))[0]
    new_name = ann_name + suffix
    new_dir = os.path.join(dir_out, new_name)

    subprocess.call(["python", path_script, "--dataroot", dir_temp, "--name", "day2night", "--model", "test", "--no_dropout", "--num_test", n, "--results_dir", new_dir], shell=True, cwd=dir_script)

    shutil.rmtree(dir_temp)
    
    dir_model = os.path.join(new_dir, "day2night")
    img_dir = os.path.join(dir_model, "test_latest", "images")
    img_paths = get_paths(img_dir)

    lookup_src = {os.path.splitext(os.path.basename(x))[0] : x for x in src.keys()}

    for img in img_paths:
        basename = os.path.basename(img)
        name = os.path.splitext(basename)[0]
        ext = os.path.splitext(basename)[1]
        if name.endswith("_fake_B"):
            name = name[:-7]
            dir_src = os.path.dirname(lookup_src[name])
            path_list = os.path.normpath(dir_src).split(os.sep)
            dir_src = os.path.join(*path_list[1:])
            tar_dir = os.path.join(new_dir, dir_src)
            tar_pth = os.path.join(tar_dir, name + ext)
            os.makedirs(tar_dir, exist_ok=True)
            if os.path.exists(tar_pth):
                os.remove(tar_pth)
            shutil.move(img, tar_pth)
    
    shutil.rmtree(dir_model)
    
    path_ann_new = os.path.join(dir_out, new_name + ".txt")
    shutil.copy(path_ann, path_ann_new)

    src_new = {}
    for img, bbs in src.items():
        path_list = os.path.normpath(img).split(os.sep)
        path_list[0] = new_name
        src_new[os.path.join(*path_list)] = bbs

    save_data(path_ann_new, src_new)

    if insert:
        for path_img, bbs in src_new.items():
            path_img_abs = os.path.join(dir_out, path_img)
            name_img = os.path.splitext(os.path.basename(path_img))[0]
            path_src = os.path.join(dir_in, lookup_src[name_img])
            img_src = cv2.imread(path_src)
            img = cv2.imread(path_img_abs)

            for b in bbs:
                x0 = int(b[0])
                y0 = int(b[1])
                x1 = int(b[2])
                y1 = int(b[3])
                img[y0:y1,x0:x1] = img_src[y0:y1,x0:x1]
            
            cv2.imwrite(path_img_abs, img)

def main():
    augment(args.method, args.path_ann, args.dir_out)

if __name__ == "__main__":
    main()