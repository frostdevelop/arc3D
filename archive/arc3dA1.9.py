import numpy as np
import pygame as pg
from numba import njit, jit, int32, float32, float64
from numba.experimental import jitclass
from numba.typed import List


#@jitclass([('verts', float32[:, :]), ("faces", uint16[:, :])])
class Object:

  def __init__(self, verts, faces, mat, **kwargs):
    self.verts = verts
    self.faces = faces
    self.mat = mat
    if mat == 1 or mat == 2:
      self.tex = pg.surfarray.array3d(pg.image.load(kwargs.get("tex", ""))).astype(np.uint8)
      self.texsize = np.array([len(self.tex) - 1,len(self.tex[0]) - 1]).astype(np.uint16)
    else:
      self.tex = np.empty((1, 3), dtype=np.uint8)
      self.texsize = np.empty(2, dtype=np.uint16)
    if mat == 2:
      self.texmap = np.array(kwargs.get("texmap",np.empty(1))).astype(np.uint16)
      self.texcoord = np.array(kwargs.get("texcoord",np.empty(1))).astype(np.float32)
    else:
      self.texmap = np.zeros((len(faces), 3)).astype(np.uint16)
      self.texcoord = np.zeros((1, 2)).astype(np.float32)


@jitclass([("position", float32[:]), ("xang", float64), ("yang", float64),("vfov", float32), ("hfov", float32), ("zfar", float32),("znear", float32)])
class Camera:

  def __init__(self, vfov, position, yang, xang, zfar, znear, width, height):
    self.position = position
    self.xang = xang
    self.yang = yang
    self.vfov = vfov
    self.zfar = zfar
    self.znear = znear


@jitclass([("dir", float32[:])])
class Light:

  def __init__(self, dir):
    self.dir = dir


class Scene:

  def __init__(self, objects, light, background):
    self.objects = objects
    self.background = np.asarray(background, dtype=np.uint8)
    self.light = light


class Renderer:

  def __init__(self, width, height, camera, uv, mouse):
    self.width = width
    self.height = height
    self.camera = camera
    self.centerx = width / 2
    self.centery = height / 2
    self.surface = np.ones((width, height, 3), dtype=np.uint8)
    self.zbuffer = np.ones((width, height), dtype=np.float32)
    self.projection = np.empty((4, 4), np.float32)
    self.projection[0, 0] = 1 / (np.tan(self.camera.vfov / 2) * (self.width / self.height))
    self.projection[1, 1] = 1 / np.tan(self.camera.vfov / 2)
    self.projection[2, 2] = self.camera.zfar / (self.camera.zfar - self.camera.znear)
    self.projection[2, 3] = 1
    self.projection[3, 2] = (-self.camera.zfar * self.camera.znear) / (
        self.camera.zfar - self.camera.znear)
    self.uv = np.asarray(uv, dtype=np.float32)
    self.mouse = mouse
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

  def proj(self, verticies):
    return self.project3(verticies, self.projection, self.centerx,
                         self.centery)

  @staticmethod
  @njit()
  def rendertris(verticies,
                 rverts,
                 tris,
                 camera,
                 light,
                 width,
                 height,
                 frame,
                 zbuffer,
                 mat,
                 tex,
                 texsize,
                 iuv=np.empty((3, 2), dtype=np.float32),
                 texmap=np.empty((1, 3), dtype=np.uint16),
                 texcoord=np.empty((1, 2), dtype=np.float32)):
    if mat == 0:
      colscale = 230 / np.max(np.abs(verticies))
    else:
      colscale = 0

    if mat == 1:
      uv = iuv
    else:
      uv = np.empty((3, 2), dtype=np.float32)
    shade = np.empty(len(tris), dtype=np.float32)
    for i in range(len(tris)):
      trii = tris[i]

      vet1 = verticies[trii[1]] - verticies[trii[0]]
      vet2 = verticies[trii[2]] - verticies[trii[0]]

      n = np.cross(vet1, vet2).astype(np.float32)
      n /= np.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])

      camRay = (verticies[trii[0]] - camera) / rverts[trii[0]][3]

      trix = [rverts[trii[0]][0], rverts[trii[1]][0], rverts[trii[2]][0]]
      triy = [rverts[trii[0]][1], rverts[trii[1]][1], rverts[trii[2]][1]]
      zmin = min([rverts[trii[0]][3], rverts[trii[1]][3], rverts[trii[2]][3]])

      if dot3d(n, camRay) < 0 and zmin > 0 and (
          (trix[0] >= -width and trix[0] <= 2 * width and triy[0] >= -height
           and triy[0] <= 2 * height) or
          (trix[1] >= -width and trix[1] <= 2 * width and triy[1] >= -height
           and triy[1] <= 2 * height) or
          (trix[2] >= -width and trix[2] <= 2 * width and triy[2] >= -height
           and triy[2] <= 2 * height)):

        shade[i] = min(1, max(0, (0.5 * dot3d(n, light.dir) + 0.5) + 0.1))
        tri = rverts[trii]
        if mat == 0:
          color = shade[i] * np.abs(verticies[trii[0]] * colscale + 25)
        else:
          color = np.empty(3)

        if mat == 2:
          uv = texcoord[texmap[i]]

        #if tri[:,0].min() < 0 or tri[:,0].max() > width or tri[:,1].min() < 0 or tri[:,1].max() > height: continue

        ysort = np.argsort(tri[:, 1])
        xstart, ystart = tri[ysort[0]][:2]
        xmiddle, ymiddle = tri[ysort[1]][:2]
        xend, yend = tri[ysort[2]][:2]

        zstart, zmiddle, zend = tri[ysort[0]][3], tri[ysort[1]][3], tri[ysort[2]][3]

        zstart, zmiddle, zend = 1 / (zstart + 1e-32), 1 / (
            zmiddle + 1e-32), 1 / (zend + 1e-32)
        zslope1 = (zend - zstart) / (yend - ystart + 1e-32)
        zslope2 = (zmiddle - zstart) / (ymiddle - ystart + 1e-32)
        zslope3 = (zend - zmiddle) / (yend - ymiddle + 1e-32)

        xslope1 = (xend - xstart) / (yend - ystart + 1e-32)
        xslope2 = (xmiddle - xstart) / (ymiddle - ystart + 1e-32)
        xslope3 = (xend - xmiddle) / (yend - ymiddle + 1e-32)

        uvstart = uv[ysort[0]] * zstart
        uvmiddle = uv[ysort[1]] * zmiddle
        uvend = uv[ysort[2]] * zend

        uvslope1 = (uvend - uvstart) / (yend - ystart + 1e-32)
        uvslope2 = (uvmiddle - uvstart) / (ymiddle - ystart + 1e-32)
        uvslope3 = (uvend - uvmiddle) / (yend - ymiddle + 1e-16)

        for y in range(max(0, ystart), min(height, yend+1)):
          yc = y - ystart
          #print(yc)
          x1 = xstart + xslope1 * yc
          uv1 = uvstart + yc * uvslope1
          z1 = zstart + zslope1 * yc

          if y < ymiddle:
            x2 = xstart + xslope2 * yc
            uv2 = uvstart + yc * uvslope2
            z2 = zstart + zslope2 * yc
          else:
            yc = y - ymiddle
            x2 = xmiddle + xslope3 * yc
            uv2 = uvmiddle + yc * uvslope3
            z2 = zmiddle + zslope3 * yc

          if x1 > x2:
            x1, x2 = x2, x1
            uv1, uv2 = uv2, uv1
            z1, z2 = z2, z1

          xc1, xc2 = max(0, min(int(x1+1),width)), max(0,min(int(x2+1), width))

          if xc1 != xc2:
            zslope = (z2 - z1) / (x2 - x1 + 1e-32)
            uvslope = (uv2 - uv1) / (x2 - x1 + 1e-32)
            for x in range(xc1, xc2):
              cx = x - x1
              z = 1 / (z1 + cx * zslope + 1e-32)
              if z < zbuffer[x, y]:
                uvo = (uv1 + cx * uvslope) * z
                if min(uvo) >= 0 and max(uvo) <= 1:
                  zbuffer[x, y] = z
                  if mat == 1 or mat == 2:
                    frame[x, y] = shade[i] * tex[int(uvo[0] * texsize[0]),int(uvo[1] * texsize[1])] * (max(0, 100 - z) / 100)
                    #frame[x, y] = tex[int(uvo[0]*texsize[0]), int(uvo[1]*texsize[1])]
                else:
                  zbuffer[x, y] = z
                  if mat == 0:
                    frame[x, y] = color
                  elif mat == 3:
                    frame[x, y] = max(0, 255 - z * 10)

  @staticmethod
  @njit()
  def project3(verticies, projection, centerx, centery):
    nverts = np.empty((len(verticies), 4), dtype=np.float32)
    for i in range(len(verticies)):
      vert = np.asarray(verticies[i])
      vert = np.append(vert, 1.0)
      nvert = matrixvect(vert, projection)
      if nvert[3] >= 0:
        nvert = np.append(np.divide(nvert[:3], nvert[3]),nvert[3] + 1e-16).astype(np.float32)
      else:
        nverts[i] = np.zeros(4, dtype=np.float32)
        continue
      nvert[1] = -nvert[1]
      nvert[0] += 1
      nvert[1] += 1
      nvert[0] *= centerx
      nvert[1] *= centery
      nvert[0] = int(nvert[0])
      nvert[1] = int(nvert[1])
      nverts[i] = nvert
    return nverts

  def render(self, scene):
    self.surface[:, :, :] = scene.background
    translateval = np.asarray(list(map(lambda x: -x, self.camera.position)),
                              dtype=np.float32)
    self.zbuffer[:, :] = 1e32
    for object in scene.objects:
      overts = self.translate(object[0].verts, object[1])
      rverts = self.proj(
          self.rotate(self.translate(overts, translateval), -self.camera.yang,
                      -self.camera.xang))
      #zsort, shade = self.sort(overts, rverts, object[0].faces, scene)
      #frame = pg.surfarray.make_surface(self.surface)
      """
      for i in np.argsort(zsort):
        if zsort[i] >= 999999: break
        color = np.asarray(shade[i]*np.abs(object[0].verts[object[0].faces[i][0]])*colscale+25).astype(np.uint8)
        triangle = np.asarray([rverts[object[0].faces[i][0]][:2], rverts[object[0].faces[i][1]][:2], rverts[object[0].faces[i][2]][:2]]).astype(np.int16)
        #color = [0,255,80]
        #pg.draw.polygon(self.surface, color, triangle)
        self.drawtriangle(self.surface, triangle, color)
      """
      if object[0].mat != 2:
        self.rendertris(overts, rverts, object[0].faces, self.camera.position,
                        scene.light, self.width, self.height, self.surface,
                        self.zbuffer, object[0].mat, object[0].tex,
                        object[0].texsize, self.uv)
      else:
        self.rendertris(overts, rverts, object[0].faces, self.camera.position,
                        scene.light, self.width, self.height, self.surface,
                        self.zbuffer, object[0].mat,
                        object[0].tex, object[0].texsize,
                        np.empty((3, 2), dtype=np.float32), object[0].texmap,
                        object[0].texcoord)

  @staticmethod
  @njit()
  def translate(verticies, position):
    nverts = np.empty(shape=(len(verticies), 3))
    for i in range(len(verticies)):
      vert = verticies[i]
      nverts[i] = [
          vert[0] + position[0], vert[1] + position[1], vert[2] + position[2]
      ]
    return nverts

  @staticmethod
  @njit()
  def rotate(verticies, yang, xang):
    nverts = np.empty(shape=(len(verticies), 3))
    for i in range(len(verticies)):
      vert = verticies[i]
      tv = [
          vert[0] * np.cos(yang) + vert[2] * np.sin(yang), vert[1],
          vert[2] * np.cos(yang) - vert[0] * np.sin(yang)
      ]
      tv = [
          tv[0], tv[1] * np.cos(xang) - tv[2] * np.sin(xang),
          tv[1] * np.sin(xang) + tv[2] * np.cos(xang)
      ]
      nverts[i] = tv
    return nverts

  def movement(self, elapsed_time):
    if self.mouse:
      if pg.mouse.get_focused():
        cursor = pg.mouse.get_pos()
        self.camera.yang += 10 * np.clip(
            (cursor[0] - self.centerx) / self.width, -0.2, 0.2)
        self.camera.xang += 10 * np.clip(
            (cursor[1] - self.centery) / self.height, -0.2, 0.2)
        pg.mouse.set_pos(self.centerx, self.centery)
    keys = pg.key.get_pressed()
    if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]: elapsed_time *= 10
    if keys[pg.K_LCTRL]: elapsed_time *= 0.25
    if keys[pg.K_w]:
      self.camera.position[0] += elapsed_time * np.sin(self.camera.yang)
      self.camera.position[2] += elapsed_time * np.cos(self.camera.yang)
    if keys[pg.K_s]:
      self.camera.position[0] -= elapsed_time * np.sin(self.camera.yang)
      self.camera.position[2] -= elapsed_time * np.cos(self.camera.yang)
    if keys[pg.K_a]:
      self.camera.position[0] -= elapsed_time * np.cos(self.camera.yang)
      self.camera.position[2] += elapsed_time * np.sin(self.camera.yang)
    if keys[pg.K_d]:
      self.camera.position[0] += elapsed_time * np.cos(self.camera.yang)
      self.camera.position[2] -= elapsed_time * np.sin(self.camera.yang)
    if keys[pg.K_e]: self.camera.position[1] += elapsed_time
    if keys[pg.K_q]: self.camera.position[1] -= elapsed_time
    if keys[pg.K_LEFT]: self.camera.yang -= elapsed_time / 2
    if keys[pg.K_RIGHT]: self.camera.yang += elapsed_time / 2
    if keys[pg.K_UP]: self.camera.xang -= elapsed_time / 2
    if keys[pg.K_DOWN]: self.camera.xang += elapsed_time / 2


@njit()
def truncate(n):
  return int(n * 10) / 10


@njit()
def dot3d(v1, v2):
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@njit()
def matrixvect(v, m):
  t = np.empty(len(v), np.float32)
  for i in range(len(v)):
    t[i] = v[0] * m[0, i] + v[1] * m[1, i] + v[2] * m[2, i] + v[3] * m[3, i]
  return t

@njit()
def intersect(vn,vp,pn,pp):
  t = dot3d(pp-vp,pn)/dot3d(vn,pn)
  pos = np.asarray((vn[0]*t+vp[0],vn[1]*t+vp[1],vn[2]*t+vp[2]), np.float32)
  return pos

#print(intersect(np.asarray([5.1,3.5,-9.2]),np.asarray([-4.9,-6.5,-5.3]),np.asarray([5,5,-4.9]),np.asarray([-1.71,20,-7.8])))
#to do clipping

def cliptri():
  pass