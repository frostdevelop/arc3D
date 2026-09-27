import numpy as np
from pathlib import Path

def readobj(name):
    verts = []
    tris = []
    texcoord = []
    texmap = []
    f = open(name)
    for line in f:
        sline = line.split()

        if len(sline) == 0:
            continue

        if sline[0] == "v":
            verts.append(sline[1:4])

        elif sline[0] == "f":
            if "/" in sline[1]:
                l1 = sline[1].split("/")
                l2 = sline[2].split("/")
                l3 = sline[3].split("/")
                tris.append([l1[0], l2[0], l3[0]])
                texmap.append([l1[1], l2[1], l3[1]])
                if len(sline) == 5:
                    l4 = sline[4].split("/")
                    tris.append([l1[0], l3[0], l4[0]])
                    texmap.append([l1[1], l3[1], l4[1]])
            else:
                tris.append(sline[1:4])
                if len(sline) == 5:
                    tris.append([sline[1], sline[3], sline[4]])
        elif sline[0] == "vt":
            texcoord.append(sline[1:3])
        else:
            continue
    f.close()
    verts = np.asarray(verts, dtype=np.float32)
    tris = np.asarray(tris, dtype=np.int32) - 1
    if len(texcoord) > 0:
        texcoord = np.asarray(texcoord, dtype=np.float32)
        texcoord[:, 1] = 1 - texcoord[:, 1]
        texmap = np.asarray(texmap, dtype=np.int32) - 1
        return verts, tris, texcoord, texmap
    else:
        return verts, tris, None, None

def clean_object_obj(object):
    object["tris"] = np.asarray(object["tris"], dtype=np.int32) - 1
    try:
        object["tex_map"] = np.asarray(object["tex_map"], dtype=np.int32) - 1
    except:
        object["tex_map"] = np.asarray([], dtype=np.int32) - 1
    
    return object
    
"""
def clean_object_obj(object):
    object["tris"] = np.asarray(object["tris"], dtype=np.int32) - 1
    if len(object["tex_coord"]) > 0:
        object["tex_coord"] = np.asarray(object["tex_coord"], dtype=np.float32)
        object["tex_coord"][:, 1] = 1 - object["tex_coord"][:, 1]
        object["tex_map"] = np.asarray(object["tex_map"], dtype=np.int32) - 1
    else:
        object["tex_coord"] = None
        object["tex_map"] = None
    return object
"""

def prune_objects_obj(objects, verts, tex_coord):
    for object in objects:
        objects[object]["verts"] = []
        vert_indices = []
        new_tris = []
        objects[object]["tex_coord"] = []
        coord_indices = []
        new_maps = []
        
        for tri in objects[object]["tris"]:
            new_tri = []
            for vert in tri:
                if vert in vert_indices:
                    new_tri.append(vert_indices.index(vert))
                else:
                    new_tri.append(len(vert_indices))
                    objects[object]["verts"].append(verts[vert])
                    vert_indices.append(vert)
            new_tris.append(new_tri)
        
        objects[object]["tris"] = np.asarray(new_tris, dtype=np.int32)
        objects[object]["verts"] = np.asarray(objects[object]["verts"], dtype=np.float32)
        
        if len(objects[object]["tex_map"]) > 0:
            for map in objects[object]["tex_map"]:
                new_map = []
                for coord in map:
                    if coord in coord_indices:
                        new_map.append(coord_indices.index(coord))
                    else:
                        new_map.append(len(coord_indices))
                        objects[object]["tex_coord"].append(tex_coord[coord])
                        coord_indices.append(coord)
                new_maps.append(new_map)
            
            objects[object]["tex_map"] = np.asarray(new_maps, dtype=np.int32)
            objects[object]["tex_coord"] = np.asarray(objects[object]["tex_coord"], dtype=np.float32)
            objects[object]["tex_coord"][:, 1] = 1 - objects[object]["tex_coord"][:, 1]
        else:
            objects[object]["tex_map"] = None
            objects[object]["tex_coord"] = None
        
    return objects

def read_mtl(name):
    materials = {}
    f = open(name)
    
    current_material = ""
    
    for line in f:
        sline = line.split()

        if len(sline) == 0:
            continue

        if sline[0] == "newmtl":
            current_material = sline[1]
            materials[current_material] = {}
        elif sline[0] == "Kd":
            materials[current_material]["diff"] = [sline[1],sline[2],sline[3]]
        elif sline[0] == "map_Kd":
            materials[current_material]["tex_diff"] = name.parent / sline[1]
        else:
            continue
    
    f.close()
    return materials

def read_obj(name):
    objects = {
        "default": {
            "tris": [],
            "tex_map": [],
            "mat": None
        }
    }
    verts = []
    tex_coord = []
    materials = None
    current_object = "default"
    path = Path(name)
    f = open(name)
    for line in f:
        sline = line.split()

        if len(sline) == 0:
            continue

        if sline[0] == "v":
            verts.append(sline[1:4])
        elif sline[0] == "f":
            if "/" in sline[1]:
                l1 = sline[1].split("/")
                l2 = sline[2].split("/")
                l3 = sline[3].split("/")
                objects[current_object]["tris"].append([l1[0], l2[0], l3[0]])
                objects[current_object]["tex_map"].append([l1[1], l2[1], l3[1]])
                if len(sline) == 5:
                    l4 = sline[4].split("/")
                    objects[current_object]["tris"].append([l1[0], l3[0], l4[0]])
                    objects[current_object]["tex_map"].append([l1[1], l3[1], l4[1]])
            else:
                objects[current_object]["tris"].append(sline[1:4])
                if len(sline) == 5:
                    objects[current_object]["tris"].append([sline[1], sline[3], sline[4]])
        elif sline[0] == "vt":
            tex_coord.append(sline[1:3])
        elif sline[0] == "o" or sline[0] == "g":
            if len(objects[current_object]["tris"]) == 0:
                del objects[current_object]
            else:
                objects[current_object] = clean_object_obj(objects[current_object])
            current_object = sline[1]
            objects[current_object] = {
                "tris": [],
                "tex_map": [],
                "mat": None
            }
        elif sline[0] == "usemtl":
            objects[current_object]["mat"] = sline[1]
        elif sline[0] == "mtllib":
            materials = read_mtl(path.parent / sline[1])
        else:
            continue
    
    f.close()
    objects[current_object] = clean_object_obj(objects[current_object])
    
    return prune_objects_obj(objects,np.asarray(verts, dtype=np.float32),np.asarray(tex_coord, dtype=np.float32)), materials 
