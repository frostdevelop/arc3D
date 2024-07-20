import numpy as np
from numba import njit, jit, int32, float64

class Renderer:
  def __init__(self, width, height, camera):
    self.width = width
    self.height = height
    self.camera = camera
    self.centerx = width/2
    self.centery = height/2
    self.projection = np.array([[(self.height/self.width)*(1/np.tan(self.camera.hfov/2)),0,0,0],
      [0,1/np.tan(self.camera.vfov/2),0,0],
      [0,0,self.camera.zfar/(self.camera.zfar-self.camera.znear),1],
      [0,0,(-self.camera.zfar*self.camera.znear)/(self.camera.zfar/self.camera.znear),0]])

class Camera:
  def __init__(self, vfov, position, yang, xang, width, height):
    self.position = np.array(position).astype(np.float64)
    self.xang = xang
    self.yang = yang
    self.vfov = vfov
    self.hfov = self.vfov*width/height
    self.zfar = 1000
    self.znear = 0.1 

def readobj(name):
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

@njit()
def matrixvect(v, m):
  t = np.zeros(len(v),np.float64)
  for i in range(len(v)):
    t[i] = v[0] * m[0,i] + v[1] * m[1,i] + v[2] * m[2,i] + v[3] * m[3,i]
  return t

verticies, triangles = readobj("teapot.obj")
verticies = np.asarray(verticies).astype(np.float32)
camera = Camera(np.pi/8,[0,0,-10],1.5,0, 640, 480)
renderer = Renderer(640,480,camera)
vert = verticies[2]
vert = np.append(vert, 1.0)
cvert = np.array([1,2,3,4])
print(matrixvect(cvert,np.array([[1,0,0,0],
                            [0,2,0,0],
                            [0,0,3,0],
                            [2,0,0,4]])))
print(vert)
print(np.dot(vert,renderer.projection))
print()