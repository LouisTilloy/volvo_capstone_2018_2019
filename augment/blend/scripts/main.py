# Based on
# https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex/1008#1008
# Might check this out? https://www.blender.org/forum/viewtopic.php?t=27419

import os
import re
import imp
import sys
import csv
import math
import random
import datetime
from mathutils import Vector

import bpy

import node_util
import file_util
import camera_util

imp.reload(node_util)
imp.reload(file_util)
imp.reload(camera_util)

class BatchConfig:
  def __init__(self, name, signs, decals, backgrounds):
    self.name = name
    self.signs = signs
    self.decals = decals
    self.backgrounds = backgrounds
    
class Sign:
  def __init__(self, tag, name, model, image):
    self.tag = tag
    self.name = name
    self.model = model
    self.image = image

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

def load_signs():
    dir = os.path.dirname(bpy.data.filepath)
    dir_signs = os.path.join(dir, "materials", "signs")
    csv_path = os.path.join(dir, "scripts", "signs.csv")
    
    signs = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            if int(row[4]) == 1:
                signs[row[0]] = Sign(row[0],row[1],row[2],os.path.join(dir_signs, row[3]))
    
    return signs

def render_scene(scene, sign, cam, pole, file_path, res_x, res_y):
    # Change sign
    sign_entry, sign_obj = node_util.change_sign(sign)
    scene.update()
    
    # Set decal on sign
    # node_util.set_decal_image()
    # Move decal
    # node_util.set_decal_translation(random.uniform(-0.2, 0.2))
    # Scale decal
    # node_util.set_decal_scale(random.uniform(0.7, 1.3))
    # Rotate decal
    # node_util.set_decal_rotation(math.radians(random.uniform(-40, 40)))
    
    # Set background
    node_util.set_bg_image()
    # Rotate background
    node_util.set_bg_rotation(random.uniform(0, 360))
    
    # Move camera to random location
    camera_util.cam_move(cam, pole, 3.0, 12.0, -60, 60, 0, 5)
    scene.update()
    # Look at the object
    camera_util.cam_look(cam, pole)
    scene.update()
    # Look away slightly from the object
    camera_util.cam_look_away(cam, sign_obj, scene, 0.8, 0.4)
    scene.update()
    
    # Get the corners of the bounding rectangle on the screen
    # Note that 0,0 is in the bottom left of the image (flipped y axis)
    x0, y0, x1, y1 = camera_util.obj_bounding_rect(scene, cam, sign_obj)
    
    # Update the bounding rectangle in the compositioning node
    node_util.set_bbox(scene, x0, y0, x1, y1, res_x, res_y)
    # Get the pixel values of the on-camera coordinates
    px0 = x0 * res_x
    py0 = y0 * res_y
    px1 = x1 * res_x
    py1 = y1 * res_y
    pw = px1 - px0
    ph = py1 - py0
    
    # Render the scene
    scene.render.filepath = file_path
    bpy.ops.render.render( write_still=True )
    
    return sign_entry.tag, px0, res_y - py1, px1, res_y - py0

def render_batch(out_dir, name, order, debug):
    date_time_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    scene = bpy.context.scene
    pole = bpy.data.objects["Pole"]
    cam = bpy.data.objects["RenderCamera"]
    signs = load_signs()
    
    #dict_list = [["path", "x0", "y0", "x1", "y1" "tag"]]
    #dict_list = []
        
    N = sum(int(val) for key, val in order.items())
    done = 0
    # Select a random sign
    #sign = signs[random.choice(list(signs))]

    new_dir = os.path.join(out_dir, name)
    #dir = os.path.dirname(bpy.data.filepath)
    #new_dir = os.path.join(dir, "output", out_dir)
    #os.makedirs(new_dir, exist_ok=True)
    #ann_path = os.path.join(new_dir, "%s.txt" % date_time_string)
    ann_path = os.path.join(out_dir, name + ".txt")
    
    for sign_tag, n in order.items():
        sign = signs[sign_tag]
    
        node_util.bbox_set_enabled(scene, debug)
        
        # Needed to rescale 2d coordinates
        render = scene.render
        res_x = render.resolution_x * render.resolution_percentage / 100
        res_y = render.resolution_y * render.resolution_percentage / 100
        
        for i in range(int(n)):
            done += 1
            print("Render %i of %i" % (done, N))
            # Set up file path to save the file
            path = "%s_%s_%i.png" % (sign_tag, date_time_string, i)
            img_path = os.path.join(new_dir, path)
            tag, x0, y0, x1, y1 = render_scene(scene, sign, cam, pole, img_path, res_x, res_y)
            with open(ann_path, "a") as f:
                f.write("%s %s,%s,%s,%s,%s\n" % (os.path.join(name, path), int(x0), int(y0), int(x1), int(y1), tag))
    
    #file_util.mkcsv(file_path, dict_list)

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    ann_path = argv[0]
    ann_name = os.path.splitext(os.path.basename(ann_path))[0]
    new_name = ann_name + "_blend"

    data = load_data(ann_path)
    out_dir = argv[1]
    
    signs = {}
    for bbs in data.values():
        for bb in bbs:
            c = str(bb[4])
            if c not in signs:
                signs[c] = 1
            else:
                signs[c] += 1

    render_batch(out_dir, new_name, signs, False)

if __name__ == "__main__":
    main()