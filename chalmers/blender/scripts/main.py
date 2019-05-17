# Based on
# https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex/1008#1008
# Might check this out? https://www.blender.org/forum/viewtopic.php?t=27419

import imp
import os
from mathutils import Vector

import bpy

import camera_util
import node_util
import file_util

imp.reload(camera_util)
imp.reload(node_util)
imp.reload(file_util)

def render_scene(scene, cam, obj, file_path, res_x, res_y):
    # Move camera to random location
    camera_util.cam_move(cam, obj, 1.0, 5.0, -60, 60, -10, 10)
    scene.update()
    # Look at the object
    camera_util.cam_look(cam, obj)
    scene.update()
    # Get the corners of the bounding rectangle on the screen
    x0, y0, x1, y1 = camera_util.obj_bounding_rect(scene, cam, obj)
    # Update the bounding rectangle in the compositioning node
    node_util.set_bbox(scene, x0, y0, x1, y1, res_x, res_y)
    # Get the pixel values of the on-camera coordinates
    px = x0 * res_x
    py = y0 * res_y
    pw = (x1 - x0) * res_x
    ph = (y1 - y0) * res_y
    # Render the scene
    scene.render.filepath = file_path
    bpy.ops.render.render( write_still=True )
    
    return 42, px, py, pw, ph

def render_batch(scene, cam, obj, name, n, debug):
    if (debug):
        node_util.bbox_set_enabled(scene, True)
    
    # Needed to rescale 2d coordinates
    render = scene.render
    res_x = render.resolution_x * render.resolution_percentage / 100
    res_y = render.resolution_y * render.resolution_percentage / 100
    
    dir = os.path.dirname(bpy.data.filepath)
    new_dir = os.path.join(dir, 'output', name)
    file_util.mkdir(new_dir)
    
    dict_list = [['tag', 'x', 'y', 'w', 'h']]
    
    for i in range(n):
        # Set up file path to save the file
        file_path = os.path.join(new_dir, 'render%i.png' % i)
        tag, x, y, w, h = render_scene(scene, cam, obj, file_path, res_x, res_y)
        dict_list.append([tag, round(x), round(y), round(w), round(h)])
    
    file_path = os.path.join(new_dir, 'annotations.csv')
    file_util.mkcsv(file_path, dict_list)
        

def main():
    scene = bpy.context.scene
    obj = bpy.data.objects['Sign']
    cam = bpy.data.objects['RenderCamera']
    render_batch(scene, cam, obj, 'test_batch', 10, False)

if __name__ == "__main__":
    main()