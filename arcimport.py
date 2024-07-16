import numpy as np

def readobj(name):
  verts = []
  tris = []
  texcoord = []
  texmap = []
  f = open(name)
  for line in f:
    sline = line.split()

    if len(sline) == 0: continue

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
  tris = np.asarray(tris, dtype=np.int16) - 1
  if len(texcoord) > 0:
    texcoord = np.asarray(texcoord, dtype=np.float32)
    texcoord[:,1] = 1 - texcoord[:,1]
    texmap = np.asarray(texmap, dtype=np.int16) - 1
    return verts, tris, texcoord, texmap
  else:
    return verts, tris, None, None