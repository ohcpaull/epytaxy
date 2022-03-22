import bpy
import numpy as np
from scipy.spatial.transform import Rotation as R

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    
    if np.linalg.norm(v1) == 0 or  np.linalg.norm(v2) == 0:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def make_arrow(name, midpoint, length=1, radius=0.2, rotation=(0,0,0), radius_ratio=3):
    
    
    
    strut = bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, 
        radius=radius, 
        depth=length, 
        location=(float(midpoint[0]),float(midpoint[1]),float(midpoint[2])), 
        rotation=np.radians(rotation)
    )
    obj1 = bpy.context.object
    obj1.name = f"strut_{name}"
    rotator = R.from_euler("xyz", rotation, degrees=True)
    cone_pos = midpoint + length/2 * unit_vector(rotator.apply([0,0,1]))
    print(unit_vector(rotator.apply([0,0,1])))
    
    
    
    cone = bpy.ops.mesh.primitive_cone_add(
        radius1=radius * radius_ratio, 
        radius2=0, 
        depth=length / radius_ratio, 
        enter_editmode=False, 
        align='WORLD', 
        location=cone_pos, 
        rotation=np.radians(rotation),
        scale=(1, 1, 1)
    )
    obj2 = bpy.context.object
    obj2.name = f"cone_{name}"
    
    bpy.context.view_layer.objects.active = obj1
    bpy.context.view_layer.objects.active = obj2

    bpy.ops.object.join()
    
    obj = bpy.context.object
    obj.name = f"{name}"
    mesh = obj.data
    mesh.name = f"mesh_{name}"

    return obj, mesh

def make_arrow_endpoints(name, start, end, radius=0.2, radius_ratio=2):
    
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    dz = float(end[2]) - float(start[2])

    
    midpoint = np.array([dx/2, dy/2, dz/2]) + np.array(start)
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    
    vec = np.array([dx, dy, dz])
    xy_vec = np.array([dx, dx, 0])
    
    x_rot = angle_between([0,0,1], vec) * np.sign(np.dot([0,0,1], vec))
    z_rot = angle_between([1,0,0], xy_vec) * np.sign(np.dot([0,1,0], xy_vec))
    rotation = np.array([x_rot, 0, z_rot])
    
    strut = bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, 
        radius=radius, 
        depth=length, 
        location=(float(midpoint[0]),float(midpoint[1]),float(midpoint[2])), 
        rotation=rotation
    )
    obj1 = bpy.context.object
    obj1.name = f"strut_{name}"
    rotator = R.from_euler("xyz", rotation, degrees=False)
    cone_pos = midpoint + length/2 * unit_vector(rotator.apply([0,0,1]))
    print(unit_vector(rotator.apply([0,0,1])))
    
    
    
    cone = bpy.ops.mesh.primitive_cone_add(
        radius1=radius * radius_ratio, 
        radius2=0, 
        depth=length / radius_ratio, 
        enter_editmode=False, 
        align='WORLD', 
        location=cone_pos, 
        rotation=rotation,
        scale=(1, 1, 1)
    )
    obj2 = bpy.context.object
    obj2.name = f"cone_{name}"
    
    bpy.context.view_layer.objects.active = obj1
    bpy.context.view_layer.objects.active = obj2

    bpy.ops.object.join()
    
    obj = bpy.context.object
    obj.name = f"{name}"
    mesh = obj.data
    mesh.name = f"mesh_{name}"

    return obj, mesh

a_pc = 3.95
make_arrow_endpoints("P1+",[0,0,0], np.array([1,1,1])*a_pc, radius=0.15)
make_arrow_endpoints("P2+",[0,0,0], np.array([-1,1,1])*a_pc, radius=0.15)
make_arrow_endpoints("P3+",[0,0,0], np.array([1,-1,1])*a_pc, radius=0.15)
make_arrow_endpoints("P4+",[0,0,0], np.array([-1,-1,1])*a_pc, radius=0.15)
make_arrow_endpoints("P1-",[0,0,0], np.array([1,1,-1])*a_pc, radius=0.15)
make_arrow_endpoints("P2-",[0,0,0], np.array([-1,1,-1])*a_pc, radius=0.15)
make_arrow_endpoints("P3-",[0,0,0], np.array([1,-1,-1])*a_pc, radius=0.15)
make_arrow_endpoints("P4-",[0,0,0], np.array([-1,-1,-1])*a_pc, radius=0.15)
