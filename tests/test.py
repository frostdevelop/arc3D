import sys
import numpy as np
import pygame as pg
from pygame.constants import K_ESCAPE, K_LEFT, K_RIGHT, K_a, K_d, K_q, K_s, K_w, K_e, K_q, K_LEFT, K_RIGHT, K_UP, K_DOWN
from pygame.locals import QUIT

width = 640
height = 480

class Object:
  def __init__(self, verts, faces, normals):
    self.verts = verts
    self.faces = faces
    self.normals = normals

class Camera:
  def __init__(self, vfov, position, yang, xang):
    self.position = np.array(position)
    self.xang = xang
    self.yang = yang
    self.vfov = vfov
    self.hfov = self.vfov*width/height
    self.zfar = 1000
    self.znear = 0.1 

class Scene:
  def __init__(self, objects, background):
    self.objects = objects
    self.background = background

class Renderer:
  def __init__(self, width, height, camera, surface):
    self.width = width
    self.height = height
    self.camera = camera
    self.centerx = width/2
    self.centery = height/2
    self.surface = surface

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

  def project2(self, verticies):
    nverts = []
    for vert in verticies:
      nvert = [0,0]
      hangle_camerapoint = np.arctan((vert[2]-self.camera.position[2])/(vert[0]-self.camera.position[0] + 1e-16))
      if abs(self.camera.position[0]+np.cos(hangle_camerapoint)-vert[0]) > abs(self.camera.position[0]-vert[0]):
        hangle_camerapoint = (hangle_camerapoint - np.pi) % (2*np.pi)

      hangle = (hangle_camerapoint-self.camera.yang)%(2*np.pi)

      if hangle > np.pi:
        hangle -= 2*np.pi

      nvert[0] = self.width*hangle/self.camera.hfov + self.centerx

      distance = np.sqrt((vert[0]-self.camera.position[0])**2 + (vert[1]-self.camera.position[1])**2 + (vert[2]-self.camera.position[2])**2)

      vangle_camerapoint = np.arcsin((self.camera.position[1]-vert[1])/distance)
      vangle = (vangle_camerapoint-self.camera.xang)%(2*np.pi)

      if vangle > np.pi:
        vangle -= 2*np.pi

      nvert[1] = self.height*vangle/self.camera.vfov + self.centery
      nverts.append(nvert)

    return nverts

  def psort(self, verticies, tris):
    ntris = np.zeros(len(tris))
    verticies = np.array(verticies)
    for i in range(len(tris)):
      tri = tris[i]

      camRay = verticies[tri[0]] - self.camera.position
      camDist = np.sqrt(camRay[0]**2 + camRay[1]**2 + camRay[2]**2)
      camRay /= camDist

      ntris[i] = -camDist
    return ntris

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


  def render(self, scene):
    self.surface.fill(scene.background)
    #translateval = [ -x for x in self.camera.position]
    for object in scene.objects:
      #rverts = self.project(self.rotate(self.translate(object.verts, translateval), -self.camera.yang, -self.camera.xang))
      rverts = self.project2(object.verts)
      #zsort = self.psort(object.verts, object.faces)
      #for i in np.argsort(zsort):
      for i in range(len(object.faces)):
        color = np.abs(object.verts[object.faces[i][0]])*45+25
        #color = [0,255,80]
        pg.draw.polygon(self.surface, color, [rverts[object.faces[i][0]], rverts[object.faces[i][1]], rverts[object.faces[i][2]]])

  def transform(self, verticies, matrix):
    pass

  def translate(self, verticies, position):
    nverts = []
    for vert in verticies:
      nverts.append([vert[0] + position[0], vert[1] + position[1], vert[2] + position[2]])
    return nverts

  def rotate(self, verticies, yang, xang):
    nverts = []
    for vert in verticies:
      tv = [vert[0] * np.cos(yang) + vert[2] * np.sin(yang), vert[1], vert[2] * np.cos(yang) - vert[0] * np.sin(yang)]
      tv = [tv[0], tv[1] * np.cos(xang) - tv[2] * np.sin(xang), tv[1] * np.sin(xang) + tv[2] * np.cos(xang)]
      nverts.append(tv)
    return nverts

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

def main():
  global width, height
  pg.init()
  screen = pg.display.set_mode((width, height), pg.FULLSCREEN)
  info = pg.display.Info()
  width = info.current_w
  height = info.current_h
  surface = pg.surface.Surface((width, height))
  clock = pg.time.Clock()
  font = pg.font.Font(pg.font.get_default_font(), 30)
  scube = Object([[-2,-2,-2],[2,2,-2],[-2,2,-2],[2,-2,-2],[-2,-2,2],[2,2,2],[-2,2,2],[2,-2,2]],[[0,2,1],[1,3,0],[4,6,5],[5,7,4],[0,4,6],[0,2,6],[3,7,5],[3,1,5],[1,2,6],[1,5,6],[0,3,7],[0,4,7]], [])
  iverts, itris = readobj("teapot.obj")
  teapot = Object(iverts, itris, [])
  scene = Scene([teapot],[50,127,200])
  camera = Camera(np.pi/4,[-10,3,0],0,0)
  renderer = Renderer(width, height, camera, surface)
  running = True
  wkey = False
  skey = False
  akey = False
  dkey = False
  qkey = False
  ekey = False
  leftkey = False
  rightkey = False
  upkey = False
  downkey = False

  pg.display.set_caption("Arc3D")
  while running:
    elapsed_time = clock.tick()*0.001

    for event in pg.event.get():
      if event.type == pg.QUIT:
        running = False
      if event.type == pg.KEYDOWN:
        if event.key == K_ESCAPE:
          running = False
        elif event.key == K_w:
          wkey = True
        elif event.key == K_s:
          skey = True
        elif event.key == K_a:
          akey = True
        elif event.key == K_d:
          dkey = True
        elif event.key == K_q:
          qkey = True
        elif event.key == K_e:
          ekey = True
        elif event.key == K_LEFT:
          leftkey = True
        elif event.key == K_RIGHT:
          rightkey = True
        elif event.key == K_UP:
          upkey = True
        elif event.key == K_DOWN:
          downkey = True
      if event.type == pg.KEYUP:
        if event.key == K_w:
          wkey = False
        elif event.key == K_s:
          skey = False
        elif event.key == K_a:
          akey = False
        elif event.key == K_d:
          dkey = False
        elif event.key == K_q:
          qkey = False
        elif event.key == K_e:
          ekey = False
        elif event.key == K_LEFT:
          leftkey = False
        elif event.key == K_RIGHT:
          rightkey = False
        elif event.key == K_UP:
          upkey = False
        elif event.key == K_DOWN:
          downkey = False

    if wkey == True:
      print(camera.position[0])
      print("w")
      camera.position[0] = camera.position[0]+0.1*np.cos(camera.yang)
      camera.position[2] = camera.position[2]+0.1*np.sin(camera.yang)
      #camera.position[0] += elapsed_time*np.cos(camera.yang)
      #camera.position[2] += elapsed_time*np.sin(camera.yang)
    if skey == True:
      camera.position[0] -= elapsed_time*np.cos(camera.yang)
      camera.position[2] -= elapsed_time*np.sin(camera.yang)
    if akey == True:
      print("a")
      camera.position[0] += elapsed_time*np.sin(camera.yang)
      camera.position[2] -= elapsed_time*np.cos(camera.yang)
    if dkey == True:
      print("d")
      camera.position[0] -= elapsed_time*np.sin(camera.yang)
      camera.position[2] += elapsed_time*np.cos(camera.yang)
    if qkey == True:
      camera.position[1] -= elapsed_time
    if ekey == True:
      camera.position[1] += elapsed_time
    if leftkey == True:
      camera.yang -= elapsed_time/10
    if rightkey == True:
      camera.yang += elapsed_time/10
    if upkey == True:
      camera.xang -= elapsed_time/10
    if downkey == True:
       camera.xang += elapsed_time/10

    scene.objects[0].verts = renderer.rotate(scene.objects[0].verts, elapsed_time/10, elapsed_time/5)

    renderer.render(scene)

    screen.blit(surface, (0, 0))

    positiontext = font.render(f'XYZ: {truncate(camera.position[0])} {truncate(camera.position[1])} {truncate(camera.position[2])}', False, (0, 0, 0))
    angletext = font.render(f'Angle: {truncate(camera.yang)} {truncate(camera.xang)}', False, (0,0,0))
    screen.blit(positiontext, (10, 10))
    screen.blit(angletext, (10, 50))
    pg.display.update()

if __name__ == "__main__":
  main()
  pg.quit()