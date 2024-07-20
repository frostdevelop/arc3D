def readobj(name):
  verts = []
  tris = []
  texmap = []
  texcoord = []
  f = open(name)
  for line in f:
    line = line.split()

    if len(line) == 0: continue

    if line[0] == "v":
      ver = list(map(float, line[1:4]))
      verts.append(ver)
    elif line[0] == "f":
      tri = list(map(lambda x: int(x)-1, line[1:4]))
      tris.append(line[1:4])
    else:
      continue

  f.close()
  return verts, tris

def readobj2(name):
  verts = []
  tris = []
  f = open(name)
  for line in f:
    if line[:2] == "v ":
      x = line.find(" ")
      y = line.find(" ", x+1)
      z = line.find(" ", y+1)
      verts.append([float(line[x+1:y]), float(line[y+1:z]), float(line[z+1:-1])])
    elif line[:2] == "f ":
      f1 = line.find(" ")
      f2 = line.find(" ", f1+1)
      f3 = line.find(" ", f2+1)
      tris.append([int(line[f1+1:f2])-1, int(line[f2+1:f3])-1, int(line[f3+1:-1])-1])
    else:
      continue

  f.close()
  return verts, tris

ver, tri = readobj2("mountains.obj")
print(tri)