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


def dist_between(p1, p2):
    """
    Returns the distance between two points in 3D space

    Parameters
    ----------
    p1 : tuple of float or np.array with shape (3,)
    p2 : tuple of float or np.array with shape (3,)

    Returns
    ----------
    dist : float
        distance between p1 and p2
    """
    xi, yi, zi = float(p1[0]), float(p1[1]), float(p1[2])
    xf, yf, zf  = float(p2[0]), float(p2[1]), float(p2[2])
    dist = np.sqrt(
        (xf - xi)**2 + (yf - yi)**2 + (zf - zi)**2
    )
    return dist


def make_strut(name, start_point, end_point, radius=1, res=16):
    """
    Creates primitive blender cylinder based off the specified start and end points
    
    Parameters
    ----------
    name : str
        Name of object
    start_point : 3-tuple of float
        3D coordinates of the start-point of the cylinder
    end_point : 3-tuple of float
        3D coordinates of the end-point of the cylinder
    radius : float, optional
        radius of cylinder
    res : int, optional
        number of vertices in the cylinder
    """ 
    (xi, yi, zi) = start_point
    (xf, yf, zf) = end_point
    cylinder_vector = (xf - xi, yf - yi, zf - zi)
    length = np.sqrt((xf - xi)**2 + (yf - yi)**2 + (zf - zi)**2)
    
    pos = ((xf - xi) / 2, (yf - yi) / 2, (zf - zi) / 2) + start_point
    
    # Rotate cylinder around y, then around z to achieve correct orientation
    y_rot = angle_between((0,0,1), cylinder_vector) # angle between Z and cylinder
    z_rot= angle_between((1,0,0), (xf - xi, yf - yi, 0)) # angle between X and projection of cylinder in XY plane

    result = bpy.ops.mesh.primitive_cylinder_add(
        vertices=res, 
        radius=radius, 
        depth=length, 
        location=(float(pos[0]),float(pos[1]),float(pos[2])), 
        rotation=(0, y_rot, z_rot)
    )#, cap_ends=True)
    obj = bpy.context.object  
    mesh = obj.data
    mesh.name = f"mesh_{name}"

    return obj, mesh

def make_arrow(name, midpoint, length=1, radius=0.2, rotation=(0,0,0), radius_ratio=3):
    """
    Make an arrowbased on the centre position and rotation
    
    Parameters
    ----------
    name : str
        Name of arrow
    midpoint : array (3,)
        Position in 3D space of arrow
    length : float
        Length of arrow
    radius : float
        Radius of arrow cylinder
    rotation : array (3,)
        Rotation of arrow in Euler XYZ rotation
    radius_ratio : float
        Ratio of the arrow cylinder to arrow head widths
    """
    
    
    strut = bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, 
        radius=radius, 
        depth=length, 
        location=(float(midpoint[0]),float(midpoint[1]),float(midpoint[2])), 
        rotation=np.radians(rotation)
    )
    obj1 = bpy.context.object
    m1 = obj1.data
    obj1.name = f"strut_{name}"

    
    # Make rotator and get the position of the cone
    rotator = R.from_euler("xyz", rotation, degrees=True)
    cone_pos = midpoint + length/2 * unit_vector(rotator.apply([0,0,1]))
    
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
    m2 = obj2.data
    obj2.name = f"cone_{name}"

    
    # Do blender magic to select both objects and join them
    bpy.ops.object.select_all(action='DESELECT')  
    obj2.select_set(True)
    obj1.select_set(True)  
    bpy.context.view_layer.objects.active = obj2
    bpy.context.view_layer.objects.active = obj1
    bpy.ops.object.join()
    
    obj = bpy.context.object
    obj.name = f"{name}"
    mesh = obj.data
    mesh.name = f"mesh_{name}"

    return obj, mesh

def make_sphere(name, radius=1., pos=None, **kw):
    res = bpy.ops.mesh.primitive_uv_sphere_add(radius=radius)#, cap_ends=True)
    obj = bpy.context.object
    obj.name = f"{name}"
    mesh = obj.data
    mesh.name = f"mesh_{name}"

    if pos is None:
        pos = np.array([0,0,0]) 
    obj.location = (float(pos[0]),float(pos[1]),float(pos[2]))

    return obj, mesh   


def make_plane(name, verts, edges, faces):
    """
    Makes a plane and does the necessary bullshit to get it working
    
    
    Parameters
    ----------
    name : str
        name of plane object
    verts : list of np.array
        the 3D vertices that you want to use to make the mesh
    edges : list of tuple
        list of tuples of the vertex indices you want to make an edge out of
    faces : list of 3-tuple
        list of tuples of vertex indices you want to make a face out of
        
    Updates
    --------
    mesh object
    """
    mesh = bpy.data.meshes.new("Plane")
    #the mesh variable is then referenced by the object variable
    #Create mesh
    # this method has an optional 'edge' array input. This is left as an empty array
    mesh.from_pydata(verts,edges,faces) 
    mesh.update(calc_edges=True) #so the edges display properly...
    
    
    myobject = bpy.data.objects.new(name, mesh)
    
    #Set location and scene of object
    myobject.location = bpy.context.scene.cursor.location # the cursor location


    return myobject, mesh

class UnitCell(object):
    
    def __init__(self, a_pc=3.905, pos=(0,0,0), stack=None, spin_rot=None):
        self.A = []
        self.B = []
        self.O = []
        self.octa = []
        self.struts = []


        self.pos=np.array(pos) * a_pc
        self.a_pc = a_pc
        self.b_pc = a_pc
        self.c_pc = a_pc
        
        if stack:
            pos = np.array(stack.pos) + np.array([0,0,stack.c_pc])
            self.pos = pos


        # Define atomic positions for A, B and O atoms
        self.A_pos = np.array([
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [1,1,0],
            [0,0,1],
            [1,0,1],
            [0,1,1],
            [1,1,1],
            ]) * a_pc + self.pos 
        self.B_pos = np.array([0.5, 0.5, 0.5]) * a_pc + self.pos 
        self.O_pos = np.array([
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0.5, 1],
        ]) * a_pc + self.pos 

        scene = bpy.context.scene
        names = [str(obj.name) for obj in list(bpy.data.objects)]

        for pos in self.A_pos:
            if (f'A_atom_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}_') in names:
                continue
            else:
                A_obj, A_mesh = make_sphere(name=f'A_atom_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}_', radius=0.5, pos=pos)
                #A_obj.select_set(True)
                self.A.append(A_obj)
        
        B_obj, B_mesh = make_sphere(name=f"B_atom_{self.B_pos}", radius=0.3, pos=self.B_pos)

        
        if spin_rot is not None:
            arr_obj, arr_mesh = make_arrow(name=f"arrow_{self.B_pos}", radius=0.1, midpoint=self.B_pos, length=3, rotation=spin_rot)
        
        
        for pos in self.O_pos:
            if (f"O_atom_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}") in names:
                continue
            else:
                O_obj, O_mesh = make_sphere(name=f"O_atom_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}", radius=0.1, pos=pos)
                self.O.append(O_obj)


        

        verts = [
            [self.O_pos[0], self.O_pos[1], self.O_pos[2]],
            [self.O_pos[0], self.O_pos[1], self.O_pos[3]],
            [self.O_pos[0], self.O_pos[2], self.O_pos[4]],
            [self.O_pos[0], self.O_pos[3], self.O_pos[4]],
            [self.O_pos[5], self.O_pos[1], self.O_pos[2]],
            [self.O_pos[5], self.O_pos[2], self.O_pos[4]],
            [self.O_pos[5], self.O_pos[1], self.O_pos[3]],
            [self.O_pos[5], self.O_pos[3], self.O_pos[4]],
        ]
        
        for vert in verts:
            tmp = make_plane(
                name=f"octahedra_{vert}",
                verts=vert,
                edges=[(0,1), (1,2), (2,0)],
                faces=[(0,1,2)]
            )
            self.octa.append(tmp)
            

            
        A_pos = [list(a) for a in self.A_pos]
        for atom in self.A_pos:
            for nn in np.array([[self.a_pc,0,0], [0, self.a_pc,0], [0,0,self.a_pc]]):
                midpt = atom + (nn / 2)  
            
                if list(atom + nn) in A_pos:
                    nearest_neighbour = (atom + nn)
                    if (f"strut_{midpt[0]:.1f}_{midpt[1]:.1f}_{midpt[2]:.1f}") in names:
                        continue
                    cyl_obj, cyl_mesh = make_strut(
                        name=f"strut_{midpt[0]:.1f}_{midpt[1]:.1f}_{midpt[2]:.1f}",
                        start_point=atom,
                        end_point=nearest_neighbour,
                        radius=0.05,
                        res=8,
                    )
                    self.struts.append(cyl_obj) 

    def add_distortion(self, dx=0, dy=0, dz=0):
        return


#objects_to_delete = list(bpy.data.objects)
#for object in objects_to_delete:
#    bpy.data.objects.remove(object, do_unlink=True)
        
for x in range(10):
    UnitCell(pos=(-x,x,0), spin_rot=(20*x,54,45),)
    UnitCell(pos=(0,-x,x), spin_rot=(20*x,35,35),)
    #UnitCell(pos=(x,0,-x), spin_rot=(0,10*x,0),)


# New Collection
O_coll = bpy.data.collections.new("O_atoms")
# Add collection to scene collection
bpy.context.scene.collection.children.link(O_coll)


# New Collection
B_coll = bpy.data.collections.new("B_atoms")
# Add collection to scene collection
bpy.context.scene.collection.children.link(B_coll)


# New Collection
A_coll = bpy.data.collections.new("A_atoms")
# Add collection to scene collection
bpy.context.scene.collection.children.link(A_coll)


# New Collection
octa_coll = bpy.data.collections.new("Octahedra")
# Add collection to scene collection
bpy.context.scene.collection.children.link(octa_coll)

# New Collection
spin_coll = bpy.data.collections.new("Spin")
# Add collection to scene collection
bpy.context.scene.collection.children.link(spin_coll)

# New Collection
strut_coll = bpy.data.collections.new("Struts")
# Add collection to scene collection
bpy.context.scene.collection.children.link(strut_coll)

object_list = list(bpy.data.objects)
to_unlink = []
for obj in object_list:
    if "O_atom" in obj.name:
        O_coll.objects.link(obj)
        to_unlink.append(obj)

    elif "B_atom" in obj.name:        
        B_coll.objects.link(obj) 
        to_unlink.append(obj)
        
    elif "A_atom" in obj.name:
        A_coll.objects.link(obj)  
        to_unlink.append(obj)

    elif "octahedra" in obj.name:
        octa_coll.objects.link(obj)
        to_unlink.append(obj)
        
    elif "arrow" in obj.name:
        spin_coll.objects.link(obj)
        to_unlink.append(obj)
        
    elif "Cylinder" in obj.name:
        strut_coll.objects.link(obj)
        to_unlink.append(obj)
        
    
    
for obj in to_unlink:
    try:
        bpy.context.scene.collection.objects.unlink(obj)
    except RuntimeError:
        pass

                


