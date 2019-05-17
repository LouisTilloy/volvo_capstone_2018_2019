import bpy

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