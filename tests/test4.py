import numpy as np
import pygame as pg
from pygame.locals import QUIT
from numba import njit, jit, int32, float32, float64
from numba.experimental import jitclass
from numba.typed import List

width = 0
height = 0
np.set_printoptions(precision=None)

#@jitclass([('verts', float32[:, :]), ("faces", uint16[:, :])])
class Object:
  def __init__(self, verts, faces, mat, **kwargs):
    self.verts = np.array(verts).astype(np.float32)
    self.faces = np.array(faces).astype(np.uint16)
    self.mat = mat
    if mat == 1 or mat == 2:
      self.tex = pg.surfarray.array3d(pg.image.load(kwargs.get("tex", None))).astype(np.uint8)
      self.texsize = [len(self.tex)-1, len(self.tex[0])-1]
    else:
      self.tex = None
      self.texsize = None
    if mat == 2:
      self.texmap = np.array(kwargs.get("texmap", None)).astype(np.uint16)
      self.texcoord = np.array(kwargs.get("texcoord", None)).astype(np.float32)
    else:
      self.texmap = None
      self.texcoord = None

@jitclass([("position", float32[:]), ("xang", float64), ("yang", float64), ("vfov", float32), ("hfov", float32), ("zfar", float32), ("znear", float32)])
class Camera:
  def __init__(self, vfov, position, yang, xang, zfar, znear, width, height):
    self.position = np.array(position).astype(np.float32)
    self.xang = xang
    self.yang = yang
    self.vfov = vfov
    self.zfar = zfar
    self.znear = znear 

@jitclass([("dir", float64[:])])
class Light:
  def __init__(self, dir):
    self.dir = np.array(dir).astype(np.float64)

class Scene:
  def __init__(self, objects, light, background):
    self.objects = objects
    self.background = np.asarray(background).astype(np.uint8)
    self.light = light

class Renderer:
  def __init__(self, width, height, camera, uv):
    self.width = width
    self.height = height
    self.camera = camera
    self.centerx = width/2
    self.centery = height/2
    frame = np.ones((width, height, 3), dtype=np.uint8)
    self.surface = frame
    self.output = pg.surfarray.make_surface(self.surface)
    self.projection = np.zeros((4,4),np.float32)
    self.projection[0,0] = 1/(np.tan(self.camera.vfov/2)*(self.width/self.height))
    self.projection[1,1] = 1/np.tan(self.camera.vfov/2)
    self.projection[2,2] = self.camera.zfar/(self.camera.zfar-self.camera.znear)
    self.projection[2,3] = 1
    self.projection[3,2] = (-self.camera.zfar*self.camera.znear)/(self.camera.zfar-self.camera.znear)
    self.uv = np.asarray(uv).astype(np.float32)
    """
    self.projection[0,0] = 1/(np.tan(self.camera.vfov/2)*(self.width/self.height))
      self.projection[1,1] = 1/np.tan(self.camera.vfov/2)
      self.projection[2,2] = -(self.camera.zfar+self.camera.znear)/(self.camera.zfar-self.camera.znear)
      self.projection[2,3] = -1
      self.projection[3,2] = -(self.camera.zfar*self.camera.znear*2)/(self.camera.zfar-self.camera.znear)
    """
    """
    self.projection = np.array([[1/(np.tan(self.camera.vfov/2)*(self.width/self.height)),0,0,0],
      [0,1/np.tan(self.camera.vfov/2),0,0],
      [0,0,self.camera.zfar/(self.camera.zfar-self.camera.znear),1],
      [0,0,-(self.camera.zfar*self.camera.znear)/(self.camera.zfar-self.camera.znear),0]])
    """

  def updatedisplay(self, width, height):
    self.width = width
    self.height = height
    self.centerx = width/2
    self.centery = height/2

  def project(self, verticies):
    rverts = []
    for vert in verticies:
      if vert[2] <= 0:
        rz = 0
      else:
        rz = self.camera.fov/vert[2]
      rverts.append([vert[0] * rz + self.centerx, vert[1] * rz + self.centery])
    return rverts

  @staticmethod
  def project2(verticies, camera, width, height):
    centerx = width/2
    centery = height/2
    nverts = np.zeros(shape=(len(verticies), 2))
    for i in range(len(verticies)):
      vert = verticies[i]
      nvert = np.zeros(2)
      hangle_camerapoint = np.arctan((vert[2]-camera.position[2])/(vert[0]-camera.position[0] + 1e-16))
      if abs(camera.position[0]+np.cos(hangle_camerapoint)-vert[0]) > abs(camera.position[0]-vert[0]):  hangle_camerapoint = (hangle_camerapoint - np.pi)%(2*np.pi)

      hangle = (hangle_camerapoint-camera.yang)%(2*np.pi)

      if hangle > np.pi: hangle -= 2*np.pi

      nvert[0] = width*hangle/camera.hfov + centerx

      distance = np.sqrt((vert[0]-camera.position[0])**2 + (vert[1]-camera.position[1])**2 + (vert[2]-camera.position[2])**2)

      vangle_camerapoint = np.arcsin((camera.position[1]-vert[1])/distance)
      vangle = (vangle_camerapoint-camera.xang)%(2*np.pi)

      if vangle > np.pi: vangle -= 2*np.pi

      nvert[1] = height*vangle/camera.vfov + centery
      nverts[i] = nvert

    return nverts

  def proj(self, verticies):
    #return self.project2(verticies, self.camera, self.width, self.height)
    return self.project3(verticies, self.projection, self.centerx, self.centery)

  def sort(self, verticies, rverts, tris, scene):
    return self.psort(np.array(verticies), rverts, tris, self.camera.position, scene.light, self.width, self.height)

  @staticmethod
  def psort(verticies, rverts, tris, camera, light, width, height):
    ntris = np.zeros(len(tris))
    shade = ntris.copy()
    for i in range(len(tris)):
      tri = tris[i]

      vet1 = verticies[tri[1]] - verticies[tri[0]]
      vet2 = verticies[tri[2]] - verticies[tri[0]]

      n = np.cross(vet1, vet2).astype(np.float32)
      n /= np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)

      camRay = verticies[tri[0]] - camera
      camDist = np.sqrt(camRay[0]**2 + camRay[1]**2 + camRay[2]**2)
      camRay /= camDist

      trix = np.asarray([rverts[tri[0]][0], rverts[tri[1]][0], rverts[tri[2]][0]])
      triy = np.asarray([rverts[tri[0]][1], rverts[tri[1]][1], rverts[tri[2]][1]])

      if dot3d(n, camRay) < 0 and np.min(trix) >= -width and np.max(trix) <= 2*width and np.min(triy) >= -height and np.max(triy) <= 2*height: 
        ntris[i] = -camDist
        shade[i] = 0.5*dot3d(n, light.dir) + 0.5
      else: ntris[i] = 999999
    return ntris, shade

  @staticmethod
  @njit()
  def rendertris(verticies, rverts, tris, camera, light, width, height, frame, zbuffer, mat, tex, texsize, *args):
    if mat == 1:
      uv = args[1]
    else:
      uv = None
    ntris = np.zeros(len(tris))
    shade = ntris.copy()
    for i in range(len(tris)):
      trii = tris[i]

      vet1 = verticies[trii[1]] - verticies[trii[0]]
      vet2 = verticies[trii[2]] - verticies[trii[0]]

      n = np.cross(vet1, vet2).astype(np.float32)
      n /= np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)

      camRay = verticies[trii[0]] - camera
      camDist = np.sqrt(camRay[0]**2 + camRay[1]**2 + camRay[2]**2)
      camRay /= camDist

      trix = np.asarray([rverts[trii[0]][0], rverts[trii[1]][0], rverts[trii[2]][0]])
      triy = np.asarray([rverts[trii[0]][1], rverts[trii[1]][1], rverts[trii[2]][1]])
      zmin = min([rverts[trii[0]][3], rverts[trii[1]][3], rverts[trii[2]][3]])

      if dot3d(n, camRay) < 0 and np.min(trix) >= -width and np.max(trix) <= 2*width and np.min(triy) >= -height and np.max(triy) <= 2*height and zmin > 0: 
        shade[i] = 0.5*dot3d(n, light.dir) + 0.5
        tri = np.asarray([list(rverts[tris[i][0]]), list(rverts[tris[i][1]]), list(rverts[tris[i][2]])]).astype(np.int16)
        if mat == 0:
          color = np.asarray(shade[i]*np.abs(tri[0,0]*args[0]+25)).astype(np.uint8)
        else:
          color = None
        #if tri[:,0].min() < 0 or tri[:,0].max() > width or tri[:,1].min() < 0 or tri[:,1].max() > height: continue
        ysort = np.argsort(tri[:, 1])
        xstart, ystart = tri[ysort[0]][:2]
        xmiddle, ymiddle = tri[ysort[1]][:2]
        xend, yend = tri[ysort[2]][:2]
        zstart = tri[ysort[0]][3]
        zmiddle = tri[ysort[1]][3]
        zend = tri[ysort[2]][3]

        xslope1 = (xend-xstart)/(yend-ystart + 1e-16)
        xslope2 = (xmiddle-xstart)/(ymiddle-ystart + 1e-16)
        xslope3 = (xend-xmiddle)/(yend-ymiddle + 1e-16)

        uvstart = uv[ysort[0]]
        uvmiddle = uv[ysort[1]]
        uvend = uv[ysort[2]]

        uvslope1 = (uvend-uvstart)/(yend-ystart + 1e-16)
        uvslope2 = (uvmiddle-uvstart)/(ymiddle-ystart + 1e-16)
        uvslope3 = (uvend-uvmiddle)/(yend-ymiddle + 1e-16)

        zslope1 = (zend-zstart)/(yend-ystart + 1e-16)
        zslope2 = (zmiddle-zstart)/(ymiddle-ystart + 1e-16)
        zslope3 = (zend-zmiddle)/(yend-ymiddle + 1e-16)

        for y in range(max(0,ystart), min(width,yend)):
          x1 = xstart + int(xslope1*(y-ystart))
          uv1 = uvstart + (y-ystart)*uvslope1
          z1 = zstart + int(zslope1*(y-ystart))

          if y < ymiddle:
            x2 = xstart + int(xslope2*(y-ystart))
            uv2 = uvstart + (y-ystart)*uvslope2
            z2 = zstart + int(zslope2*(y-ystart))
          else:
            x2 = xmiddle + int(xslope3*(y-ymiddle))
            uv2 = uvmiddle + (y-ymiddle)*uvslope3
            z2 = zmiddle + int(zslope3*(y-ymiddle))

          if x1 > x2:
            x1, x2 = x2, x1
            uv1, uv2 = uv2, uv1
            z1, z2 = z2, z1

          uvslope = (uv2-uv1)/(x2-x1 + 1e-16)

          x1, x2 = max(0,min(x1, width-1)), max(0,min(x2, width-1))

          """
          if x1 < 0: x1 = 0
          if x2 > width: x2 = width
          if y < 0: y = 0
          if y > height: y = height
          """

          for x in range(x1, x2):
            uvo = uv1 + (x-x1)*uvslope
            z = z1 + (z2 - z1)*(x - x1)/(x2 - x1 + 1e-16)
            if z < zbuffer[x, y]:
              zbuffer[x, y] = z
              if mat == 1:
                frame[x, y] = shade[i]*tex[int(uvo[0]*texsize[0]), int(uvo[1]*texsize[1])]*(max(0, 100-z)/100)
              elif mat == 0:
                frame[x1:x2,y] = color
              elif mat == 3:
                frame[x, y] = max(0, 100-z)


  def perspective(self, verticies):
    aspect = self.height/self.width
    nverts = []
    for vert in verticies:
      nvert = [0,0,0,0]
      nvert[0] = vert[0]*(aspect*(1/np.tan(self.camera.vfov/2)))
      nvert[1] = vert[1]*(1/np.tan(self.camera.vfov/2))
      nvert[2] = vert[2]*((-self.camera.zfar*self.camera.znear)/(self.camera.zfar-self.camera.znear))
      nvert[3] = vert[2]
      nverts.append(nverts)
    return nverts

  def projectp(self, verticies):
    nverts = []
    for vert in verticies:
      nvert = [0,0,0,0]
      if vert[3] != 0:
        nvert[0] = vert[0]/vert[3]
        nvert[1] = vert[1]/vert[3]
        nvert[2] = vert[2]/vert[3]
        nvert[3] = vert[3]
      nverts.append(nvert)
    return nverts

  @staticmethod
  @njit()
  def project3(verticies, projection, centerx, centery):
    nverts = np.zeros((len(verticies),4), dtype=np.float32)
    for i in range(len(verticies)):
      vert = np.asarray(verticies[i])
      vert = np.append(vert, 1.0)
      nvert = matrixvect(vert, projection)
      if nvert[3] >= 0:
        nvert = np.append(np.divide(nvert[:3],nvert[3]), nvert[3] + 1e-16).astype(np.float32)
      else:
        nvert = np.zeros(4, dtype=np.float32)
      nvert[1] = -nvert[1]
      if i < 4:
        print(i)
        print(List(nvert))
        print(List(vert))
      nvert[0] += 1
      nvert[1] += 1
      nvert[0] *= centerx
      nvert[1] *= centery
      nverts[i] = nvert
    return nverts

  def renderpoints(self, scene):
    frame = self.surface.copy()
    frame[:,:,0], frame[:,:,1], frame[:,:,2] = scene.background
    for obj in scene.objects:
      tverts = self.translate(obj[0].verts, obj[1])
      rverts = self.proj(tverts)
      for vert in rverts: frame[vert[0],vert[1]] = [255,0,0]
    self.output = surface

  def render(self, scene):
    frame = self.surface.copy()
    frame[:,:,0], frame[:,:,1], frame[:,:,2] = scene.background
    translateval = [ -x for x in self.camera.position]
    zbuffer = np.ones((width, height)) + 999999
    for object in scene.objects:
      if object[0].mat == 0:
        colscale = 230/np.max(np.abs(object[0].verts))
      else:
        colscale = 0
      overts = self.translate(object[0].verts, object[1])
      tverts = self.rotate(self.translate(overts, translateval), -self.camera.yang, -self.camera.xang)
      rverts = self.proj(tverts)
      #zsort, shade = self.sort(overts, rverts, object[0].faces, scene)
      #frame = pg.surfarray.make_surface(frame)
      """
      for i in np.argsort(zsort):
        if zsort[i] >= 999999: break
        color = np.asarray(shade[i]*np.abs(object[0].verts[object[0].faces[i][0]])*colscale+25).astype(np.uint8)
        triangle = np.asarray([rverts[object[0].faces[i][0]][:2], rverts[object[0].faces[i][1]][:2], rverts[object[0].faces[i][2]][:2]]).astype(np.int16)
        #color = [0,255,80]
        #pg.draw.polygon(frame, color, triangle)
        self.drawtriangle(frame, triangle, color)
      """
      self.rendertris(overts, rverts, object[0].faces, self.camera.position, scene.light, self.width, self.height, frame, zbuffer, object[0].mat, object[0].tex, object[0].texsize, colscale, self.uv)
    self.output = frame

  @staticmethod
  @njit()
  def translate(verticies, position):
    nverts = np.zeros(shape=(len(verticies),3))
    for i in range(len(verticies)):
      vert = verticies[i]    
      nverts[i] = [vert[0] + position[0], vert[1] + position[1], vert[2] + position[2]]
    return nverts

  @staticmethod
  @njit()
  def rotate(verticies, yang, xang):
    nverts = np.zeros(shape=(len(verticies),3))
    for i in range(len(verticies)):
      vert = verticies[i]
      tv = [vert[0] * np.cos(yang) + vert[2] * np.sin(yang), vert[1], vert[2] * np.cos(yang) - vert[0] * np.sin(yang)]
      tv = [tv[0], tv[1] * np.cos(xang) - tv[2] * np.sin(xang), tv[1] * np.sin(xang) + tv[2] * np.cos(xang)]
      nverts[i] = tv
    return nverts

  def movement(self, elapsed_time):
    keys = pg.key.get_pressed()
    if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]: elapsed_time *= 5
    if keys[pg.K_LCTRL]: elapsed_time *= 0.5
    if keys[pg.K_w]: 
      self.camera.position[0] += elapsed_time*np.sin(self.camera.yang)
      self.camera.position[2] += elapsed_time*np.cos(self.camera.yang)
    if keys[pg.K_s]:
      self.camera.position[0] -= elapsed_time*np.sin(self.camera.yang)
      self.camera.position[2] -= elapsed_time*np.cos(self.camera.yang)
    if keys[pg.K_a]:
      self.camera.position[0] -= elapsed_time*np.cos(self.camera.yang)
      self.camera.position[2] += elapsed_time*np.sin(self.camera.yang)
    if keys[pg.K_d]:
      self.camera.position[0] += elapsed_time*np.cos(self.camera.yang)
      self.camera.position[2] -= elapsed_time*np.sin(self.camera.yang)
    if keys[pg.K_e]: self.camera.position[1] += elapsed_time
    if keys[pg.K_q]: self.camera.position[1] -= elapsed_time
    if keys[pg.K_LEFT]: self.camera.yang -= elapsed_time/2
    if keys[pg.K_RIGHT]: self.camera.yang += elapsed_time/2
    if keys[pg.K_UP]: self.camera.xang -= elapsed_time/2
    if keys[pg.K_DOWN]: self.camera.xang += elapsed_time/2

  @staticmethod
  def drawtriangle(frame, tri, color):
    if tri[:,0].min() < 0 or tri[:,0].max() > frame.shape[0] or tri[:,1].min() < 0 or tri[:,1].max() > frame.shape[1]: return
    ysort = np.argsort(tri[:, 1])
    xstart, ystart = tri[ysort[0]]
    xmiddle, ymiddle = tri[ysort[1]]
    xend, yend = tri[ysort[2]]

    xslope1 = (xend-xstart)/(yend-ystart + 1e-16)
    xslope2 = (xmiddle-xstart)/(ymiddle-ystart + 1e-16)
    xslope3 = (xend-xmiddle)/(yend-ymiddle + 1e-16)

    for y in range(ystart, yend):
      x1 = xstart + int(xslope1*(y-ystart))
      if y < ymiddle:
        x2 = xstart + int(xslope2*(y-ystart))
      else:
        x2 = xmiddle + int(xslope3*(y-ymiddle))

      if x1 > x2:
        x1, x2 = x2, x1

      frame[x1:x2,y] = color

@njit()
def truncate(n):
  return int(n * 10) / 10

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
def dot3d(v1, v2):
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

@njit()
def matrixvect(v, m):
  t = np.zeros(len(v),np.float32)
  for i in range(len(v)):
    t[i] = v[0] * m[0,i] + v[1] * m[1,i] + v[2] * m[2,i] + v[3] * m[3,i]
  return t

def main():
  global width, height
  pg.init()
  screen = pg.display.set_mode((width, height), pg.FULLSCREEN)
  info = pg.display.Info()
  width = info.current_w
  height = info.current_h
  print(width, height)
  clock = pg.time.Clock()
  font = pg.font.Font(pg.font.get_default_font(), 30)
  scube = Object([[-2,-2,-2],[2,2,-2],[-2,2,-2],[2,-2,-2],[-2,-2,2],[2,2,2],[-2,2,2],[2,-2,2]],[[0,2,3],[2,1,3],[7,5,4],[5,6,4],[4,6,0],[6,2,0],[3,1,7],[1,5,7],[7,4,0],[0,3,7],[2,6,1],[6,5,1]], 1, tex="box.jpeg", texcoord=[[0,0],[0,1],[1,0],[1,1]], texmap=[[0,1,2],[1,3,2],[0,1,2],[1,3,2],[0,1,2],[1,3,2],[0,1,2],[1,3,2],[0,1,2],[1,3,2],[0,1,2],[1,3,2]])
  light = Light([0,1,1])
  scene = Scene([[scube,[0,0,0]]], light, [50,127,200])
  camera = Camera(np.pi/1.5,[0.0,0.0,-8.0],0,0,1000,0.1,width,height)
  renderer = Renderer(width, height, camera, [[1,0], [0,1], [1,1]])
  running = True

  pg.display.set_caption("Arc3D")
  while running:
    for event in pg.event.get():
      if event.type == pg.QUIT:
        running = False
      if event.type == pg.KEYDOWN:
        if event.key == pg.K_ESCAPE:
          running = False
    renderer.render(scene)
    surface = pg.surfarray.make_surface(renderer.output)
    screen.blit(surface, (0, 0))
    """
    positiontext = font.render(f'XYZ: {truncate(camera.position[0])} {truncate(camera.position[1])} {truncate(camera.position[2])}', False, (0, 0, 0))
    angletext = font.render(f'Angle: {truncate(camera.yang)} {truncate(camera.xang)}', False, (0,0,0))
    screen.blit(positiontext, (10, 10))
    screen.blit(angletext, (10, 50))
    """
    pg.display.update()

if __name__ == "__main__":
  main()
  pg.quit()