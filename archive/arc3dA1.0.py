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
  def __init__(self, fov, position, yang, xang):
    self.position = position
    self.xang = xang
    self.fov = fov
    self.yang = yang
    self.vfov = np.pi/4
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

  def render(self, scene):
    self.surface.fill(scene.background)
    translateval = [ -x for x in self.camera.position]
    for object in scene.objects:
      rverts = self.project(self.rotate(self.translate(object.verts, translateval), -self.camera.yang, -self.camera.xang))
      for face in object.faces:
        color = [0,255,80]
        pg.draw.polygon(self.surface, color, [rverts[face[0]], rverts[face[1]], rverts[face[2]]])

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
  model = Object([[-50,0,0],[-50,0,100],[50,0,0]],[[0,1,2]],[])
  cube = Object([[-150,-150,-150],[150,150,-150],[-150,150,-150],[150,-150,-150],[-150,-150,150],[150,150,150],[-150,150,150],[150,-150,150]],[[0,2,1],[1,3,0],[4,6,5],[5,7,4],[0,4,6],[0,2,6],[3,7,5],[3,1,5],[1,2,6],[1,5,6],[0,3,7],[0,4,7]], [])
  scene = Scene([cube],[50,127,200])
  camera = Camera(300,[0,-100,-500],0,0)
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
      camera.position[0] += 1.1*np.sin(camera.yang)
      camera.position[2] += 1.1*np.cos(camera.yang)
    if skey == True:
      camera.position[0] -= 1.1*np.sin(camera.yang)
      camera.position[2] -= 1.1*np.cos(camera.yang)
    if akey == True:
      camera.position[0] -= 1.1*np.cos(camera.yang)
      camera.position[2] += 1.1*np.sin(camera.yang)
    if dkey == True:
      camera.position[0] += 1.1*np.cos(camera.yang)
      camera.position[2] -= 1.1*np.sin(camera.yang)
    if qkey == True:
      camera.position[1] += 1.1
    if ekey == True:
      camera.position[1] -= 1.1
    if leftkey == True:
      camera.yang -= 0.005
    if rightkey == True:
      camera.yang += 0.005
    if upkey == True:
      camera.xang += 0.005
    if downkey == True:
       camera.xang -= 0.005

    cube.verts = renderer.rotate(cube.verts, 0.001, 0.002)

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