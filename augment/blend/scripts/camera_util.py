import math
import random
import mathutils
from mathutils import Vector

import bpy
from bpy_extras.object_utils import world_to_camera_view

# https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def cam_look(cam, obj):
    loc_obj = obj.matrix_world.to_translation()
    loc_cam = cam.matrix_world.to_translation()

    dir = loc_obj - loc_cam
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = dir.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    cam.rotation_euler = rot_quat.to_euler()
    
def cam_move(cam, obj, r_min, r_max, theta_min, theta_max, phi_min, phi_max):
    # https://blender.stackexchange.com/questions/14137/converting-global-object-location-to-local-location-in-python
    r = random.uniform(r_min, r_max)
    theta = math.radians(random.uniform(theta_min, theta_max))
    phi = math.radians(random.uniform(phi_min, phi_max))
    
    obj_loc_local = obj.matrix_world.inverted() * obj.location
    
    new_loc = Vector()
    new_loc.x = obj_loc_local.x + r * math.cos(phi) * math.cos(theta)
    new_loc.y = obj_loc_local.y - r * math.cos(phi) * math.sin(theta)
    new_loc.z = obj_loc_local.z + r * math.sin(phi)
    
    cam.location = obj.matrix_world * new_loc
    
def obj_bounding_rect(scene, cam, obj):
    # use generator expressions () or list comprehensions []
    # Had problem with negative y-coord - added obj.matrix_world from
    # https://www.reddit.com/r/blender/comments/4s8k3f/help_with_finding_whether_a_vertex_is_visible/
    
    obj_mat_world = obj.matrix_world
    verts = (obj_mat_world * vert.co for vert in obj.data.vertices)
    coords_2d = [world_to_camera_view(scene, cam, coord) for coord in verts]

    # 2d data printout:
    # rnd = lambda i: round(i)
    # for x, y, distance_to_lens in coords_2d:
    #     print("{},{}".format(rnd(res_x*x), rnd(res_y*y)))

    x0 = coords_2d[0].x
    y0 = coords_2d[0].y
    x1 = x0
    y1 = y0

    # print('x,y')
    for x, y, distance_to_lens in coords_2d:
        if (x < x0): x0 = x
        if( y < y0): y0 = y
        if (x > x1): x1 = x
        if (y > y1): y1 = y

    return x0, y0, x1, y1

def cam_look_away(cam, obj, scene, x_factor, y_factor):
    cam_data = bpy.data.cameras['RenderCameraData']
    
    fov_x = 0.0
    fov_y = 0.0
    ratio = scene.render.resolution_x * 1.0 / scene.render.resolution_y
    
    if ratio > 1.0:
        fov_x = cam_data.angle
        fov_y = 2.0 * math.atan(1.0 / ratio * math.tan(fov_x / 2.0))
    else:
        fov_y = cam_data.angle
        fov_x = 2.0 * math.atan(ratio * math.tan(fov_y / 2.0))
    
    obj_loc_local = cam.matrix_world.inverted() * obj.matrix_world.to_translation()
    verts_local = [cam.matrix_world.inverted() * obj.matrix_world * vert.co for vert in obj.data.vertices]
    
    # Camera is pointing in negative z-direction locally
    angles_x = [math.atan(v.x / abs(v.z)) for v in verts_local]
    angles_y = [math.atan(v.y / abs(v.z)) for v in verts_local]
    
    # Need to find max in X and Y directions respectiveley
    
    max_x = angles_x[0]
    min_x = angles_x[0]
    max_y = angles_y[0]
    min_y = angles_y[0]
    
    for a in angles_x:
        if (a > max_x): max_x = a
        if (a < min_x): min_x = a
        
    for a in angles_y:
        if (a > max_y): max_y = a
        if (a < min_y): min_y = a
    
    # Yaw
    yaw_rot = x_factor * random.uniform(-fov_x / 2.0 + max_x, fov_x / 2.0 + min_x)
    cam.rotation_euler.rotate_axis("Y", yaw_rot)
    
    # Pitch
    pitch_rot = y_factor * random.uniform(-fov_y / 2.0 + max_y, fov_y / 2.0 + min_y)
    cam.rotation_euler.rotate_axis("X", pitch_rot)