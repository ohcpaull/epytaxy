import bpy
import numpy as np
from math import sqrt



def make_cylinder(name,radius=1.,length=1.,res=16,pos = (0,0,0), rot=(0,0,0)):

    res = bpy.ops.mesh.primitive_cylinder_add(vertices=res, radius=radius, 
                    depth=length, rotation=rot)#, cap_ends=True)
    obj = bpy.context.object
    obj.name = name
    mesh = obj.data
    if pos is None:
        pos = np.array([0,0,0])
        
    obj.name = f"{name}_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}"

    obj.location = (float(pos[0]),float(pos[1]),float(pos[2]))

    return obj, mesh


def make_sphere(name, radius=1., pos=None, **kw):
    res = bpy.ops.mesh.primitive_uv_sphere_add(radius=radius)#, cap_ends=True)
    obj = bpy.context.object
    obj.name = f"{name}_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}"
    mesh = obj.data
    mesh.name = "mesh_"+name

    if pos is None:
        pos = np.array([0,0,0]) 
    obj.location = (float(pos[0]),float(pos[1]),float(pos[2]))

    return obj, mesh   

def make_plane(name, verts, edges, faces):
    """
    Makes a plane and does the necessary things to get it working
    
    
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
    myobject = bpy.data.objects.new(name, mesh)
    
    #Set location and scene of object
    myobject.location = bpy.context.scene.cursor.location # the cursor location
    bpy.context.scene.collection.objects.link(myobject) # linking the object to the scene

    #Create mesh
    # this method has an optional 'edge' array input. This is left as an empty array
    mesh.from_pydata(verts,edges,faces) 
    mesh.update(calc_edges=True) #so the edges display properly...
    return myobject


class UnitCell(object):
    
    def __init__(self, a_pc=3.91, c_pc = 4.19, pos=(0,0,0), stack=None):
        
        self.a_pc = a_pc
        self.b_pc = a_pc
        self.c_pc = c_pc
        
        self.pos=np.array(pos) * [a_pc, a_pc, c_pc]
        
        
        if stack:
            pos = stack.pos + np.array([0,0,stack.c_pc])
            self.pos = pos
        
        # Add blender collections for each atom type
        self.A = bpy.data.collections.new("A_atoms")
        self.B = bpy.data.collections.new("B_atoms")
        self.O = bpy.data.collections.new("O_atoms")
        self.struts = bpy.data.collections.new("struts")
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
            ]) * [a_pc, a_pc, c_pc] + self.pos 
        self.B_pos = np.array([0.5, 0.5, 0.5]) * [a_pc, a_pc, c_pc] + self.pos 
        self.O_pos = np.array([
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0.5, 1],
        ]) * [a_pc, a_pc, c_pc] + self.pos 

        scene = bpy.context.scene

        # Create atom in A positions and add to separate collections
        for pos in self.A_pos:
            if (f"A_atom_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}") in bpy.data.objects:
                pass
            else:
                A_obj, A_mesh = make_sphere(name=f"A_atom_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}", radius=0.5, pos=pos)
                #A_obj.select_set(True)
                bpy.data.collections["A_atoms"].objects.link(A_obj)


        B_obj, B_mesh = make_sphere(name=f"B_atom_{self.B_pos}", scale=0.3, pos=self.B_pos)
        B_obj.select_set(True)
        bpy.data.collections["B_atoms"].objects.link(B_obj)
        
        for pos in self.O_pos:
            if (f"O_atom_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}") in bpy.data.objects:
                pass
            else:
                O_obj, O_mesh = make_sphere(name=f"O_atom_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}", radius=0.1, pos=pos)
                bpy.data.collections["O_atoms"].objects.link(O_obj)

        self.octahedra = bpy.data.collections.new("octahedra")

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
            bpy.data.collections["octahedra"].objects.link(tmp)

        # Make cylinders between A-atoms
        cyl_pos = np.array([
            [0.5,0,0],
            [0,0.5,0],
            [1,0.5,0],
            [0.5,1,0],
            [0,0,0.5],
            [1,0,0.5],
            [0,1,0.5],
            [1,1,0.5],
            [0.5,0,1],
            [0,0.5,1],
            [1,0.5,1],
            [0.5,1,1],
            ]) * [a_pc, a_pc, c_pc] + self.pos 
            
        cyl_ang = [
            (0,np.pi/2,0),
            (np.pi/2,0,0),
            (np.pi/2,0,0),
            (0,np.pi/2,0),
            (0,0,0),
            (0,0,0),
            (0,0,0),
            (0,0,0),
            (0,np.pi/2,0),
            (np.pi/2,0,0),
            (np.pi/2,0,0),
            (0,np.pi/2,0),
        ]

        for pos, angle in zip(cyl_pos, cyl_ang):
            cyl_obj, cyl_mesh = make_cylinder(
                name=f"strut__{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}",
                length=a_pc, 
                radius=0.05,
                pos=pos,
                rot=angle,
            )
            bpy.data.collections["struts"].objects.link(cyl_obj)




objects_to_delete = list(bpy.data.objects)
for object in objects_to_delete:
    bpy.data.objects.remove(object, do_unlink=True)

cells_A = []
cells_B = []
ystep = 0

for x in range(2):
    for y in range(2):
        for z in range(2):
            cells_A.append(UnitCell(a_pc=3.79, c_pc=3.79, pos=(x,y,z)))
            

for idx, cell in enumerate(cells_A):
    if cell.pos[2] > 0.5:
        cells_B.append(UnitCell(a_pc = 3.79, c_pc= 4.67, stack=cells_A[idx]))
        cells_B.append(UnitCell(a_pc = 3.79, c_pc= 4.67, stack=cells_B[-1]))



    
