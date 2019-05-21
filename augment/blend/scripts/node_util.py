import bpy
import os
import random
from random import randint
import csv

def bbox_set_enabled(scene, v):
    box_mask = scene.node_tree.nodes['BoxMask']
    box_mask.inputs[1].default_value = 1 if v else 0

def set_bbox(scene, x0, y0, x1, y1, res_x, res_y):
    # https://blender.stackexchange.com/questions/40176/how-to-introspectively-modify-nodes-from-python
    box_mask = scene.node_tree.nodes['BoxMask']
    
    # Middle of the box
    box_mask.x = (x0 + x1) / 2
    box_mask.y = (y0 + y1) / 2
    
    # The camera will use 1.0 as the largest of the width or height
    if (res_x > res_y):
        box_mask.width = x1 - x0
        box_mask.height = (y1 - y0) * res_y / res_x
    else:
        box_mask.width = (x1 - x0) * res_x / res_y
        box_mask.height = y1 - y0

def set_decal_image():
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    dir = os.path.dirname(bpy.data.filepath)
    decal_dir = os.path.join(dir, 'materials', 'decals', 'graffiti')
    
    onlyfiles = [f for f in os.listdir(decal_dir) if os.path.isfile(os.path.join(decal_dir, f))]
    
    i = randint(0, len(onlyfiles) - 1)
    img_path = os.path.join(decal_dir, onlyfiles[i])
    
    # Load image from file
    img = bpy.data.images.load(img_path)
    
    mat = bpy.data.materials['Sign']
    nodes = mat.node_tree.nodes
    decal_texture = nodes.get('DecalTexture')
    
    # Set the image of the node
    decal_texture.image = img
        
def set_decal_rotation(x):
    # https://blender.stackexchange.com/questions/23436/control-cycles-material-nodes-and-material-properties-in-python
    mat = bpy.data.materials['Sign']
    nodes = mat.node_tree.nodes
    mapping = nodes.get('MappingRotation')
    
    # Rotation around Z axis
    mapping.rotation[2] = x
    
def set_decal_scale(x):
    mat = bpy.data.materials['Sign']
    nodes = mat.node_tree.nodes
    mapping = nodes.get('MappingScale')
    
    # Scale both the X and Y axis. The scaling is inverted in the node editor
    mapping.scale[0] = 1.0 / x
    mapping.scale[1] = 1.0 / x
    
def set_decal_translation(x):
    mat = bpy.data.materials['Sign']
    nodes = mat.node_tree.nodes
    mapping = nodes.get('MappingTranslation')
    
    # Translate along both the X and Y axis.
    mapping.translation[0] = x
    mapping.translation[1] = x
    
def set_bg_image():
    dir = os.path.dirname(bpy.data.filepath)
    bg_dir = os.path.join(dir, 'materials', 'backgrounds', 'night')
    onlyfiles = [f for f in os.listdir(bg_dir) if os.path.isfile(os.path.join(bg_dir, f))]
    
    i = randint(0, len(onlyfiles) - 1)
    img_path = os.path.join(bg_dir, onlyfiles[i])
    
    # Load image from file
    img = bpy.data.images.load(img_path)
    
    # Set the image of the node
    env = bpy.data.worlds['World'].node_tree.nodes.get('Environment Texture')
    env.image = img
    
def set_bg_rotation(x):
    world = bpy.data.worlds['World']
    nodes = world.node_tree.nodes
    mapping = nodes.get('Mapping')
    
    mapping.rotation[2] = x
    
def change_sign(sign):
    # Hide all signs
    bpy.data.objects['SignOctagon'].hide_render = True
    bpy.data.objects['SignHouse'].hide_render = True
    bpy.data.objects['SignTallRectangle'].hide_render = True
    bpy.data.objects['SignRectangle'].hide_render = True
    bpy.data.objects['SignSquare'].hide_render = True
    bpy.data.objects['SignTriangle'].hide_render = True
    bpy.data.objects['SignInvertedTriangle'].hide_render = True
    bpy.data.objects['SignDiamond'].hide_render = True
    bpy.data.objects['SignCircle'].hide_render = True
    
    sign_obj = bpy.data.objects[sign.model]
    sign_obj.hide_render = False
    
    # Set the image of the signtry:
    try:
        img = bpy.data.images.load(sign.image)
    except:
        img = bpy.data.images['Error']
        
    nodes = bpy.data.materials['Sign'].node_tree.nodes
    nodes.get('SignTexture').image = img
        
    return sign, sign_obj