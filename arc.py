import numpy as np
import math
import pygame as pg
from numba import njit, jit, int32, float32, float64
from numba.experimental import jitclass

#@jitclass([('verts', float32[:, :]), ("faces", uint16[:, :])])
class Object:
  def __init__(self, verts, faces, mat, pos, rot, **kwargs):
    self.mverts = np.asarray(verts, np.float32)
    self.faces = np.asarray(faces, np.uint16)
    self.flen = len(self.faces)
    self.vlen = len(self.mverts)
    self.position = np.asarray(pos, np.float32)
    self.rotation = np.asarray(rot, np.float32)
    self.upd()
    self.mat = mat
    match mat:
      case 1 | 2:
        self.tex = pg.surfarray.array3d(pg.image.load(kwargs.get(
            "tex", ""))).astype(np.uint8)
        self.texsize = np.array([len(self.tex) - 1,len(self.tex[0]) - 1]).astype(np.uint16)
      case _:
        self.tex = np.empty((1, 3), dtype=np.uint8)
        self.texsize = np.empty(2, dtype=np.uint16)
    match mat:
      case 2:
        self.texmap = np.array(kwargs.get("texmap",  np.empty(1))).astype(np.uint32)
        self.texcoord = np.array(kwargs.get("texcoord",         np.empty(1))).astype(np.float32)
        self.clen = len(self.texcoord)
      case 1:
        self.texmap = np.zeros((len(faces), 3)).astype(np.uint32)
        self.texcoord = np.zeros((1, 2)).astype(np.float32)
        self.clen = 3
      case _:
        self.texmap = np.zeros((len(faces), 3)).astype(np.uint32)
        self.texcoord = np.zeros((1, 2)).astype(np.float32)
        self.clen = 0

  def upd(self):
    self.verts = Renderer.rotate(self.mverts, self.rotation[0], self.rotation[1])
    self.verts = Renderer.translate(self.verts, self.position)

@jitclass([("position", float32[:]), ("angle", float32[:]),
           ("vfov", float32), ("hfov", float32), ("zfar", float32),
           ("znear", float32), ("velocity", float32[:]), ("speed", float32)])
class Camera:
  #replace angle with two vars?
  def __init__(self, vfov, position, angle, zfar, znear):
    self.position = np.asarray(position, np.float32)
    self.angle = np.asarray(angle, np.float32)
    self.vfov = vfov
    self.zfar = zfar
    self.znear = znear
    self.velocity = np.zeros(3,np.float32)
    self.speed = 10


@jitclass([("dir", float32[:])])
class Light:

  def __init__(self, dir):
    self.dir = np.asarray(dir,np.float32)
    self.dir = self.dir / math.sqrt(self.dir[0]*self.dir[0] + self.dir[1]*self.dir[1] + self.dir[2]*self.dir[2])

  def upd(self, ndir):
    self.dir = np.asarray(ndir,np.float32)
    self.dir = self.dir / math.sqrt(self.dir[0]*self.dir[0] + self.dir[1]*self.dir[1] + self.dir[2]*self.dir[2])


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
    self.centerx = width >> 1
    self.centery = height >> 1
    self.surface = np.ones((width, height, 3), dtype=np.uint8)
    self.nearplane = np.asarray(((0, 0, camera.znear), (0, 0, 1)), np.float32)
    self.zbuffer = np.empty((width, height), dtype=np.float32)
    self.projection = np.zeros((4, 4), np.float32)
    self.projection[0, 0] = 1 / (np.tan(self.camera.vfov / 2) * (self.width / self.height))
    self.projection[1, 1] = -(1 / np.tan(self.camera.vfov / 2))
    self.projection[2, 2] = self.camera.zfar / (self.camera.zfar - self.camera.znear)
    self.projection[2, 3] = 1
    self.projection[3, 2] = (-self.camera.zfar * self.camera.znear) / (self.camera.zfar - self.camera.znear)
    self.uv = np.asarray(uv, dtype=np.float32)
    self.mouse = mouse

  @staticmethod
  @njit()
  def rendertris(bk,
                 verticies,
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
                 texmap,
                 texcoord):
    match mat:
      case 0:
        colscale = 230 / max(np.abs(verticies))

    uv = np.empty((3, 2), np.float32)
    for i in range(len(tris)):
      tri = rverts[tris[i]]
      if (tri[0][0] >= -width and tri[0][0] <= width << 1 and tri[0][1] >= -height and tri[0][1] <= height << 1) or (tri[1][0] >= -width and tri[1][0] <= width << 1 and tri[1][1] >= -height and tri[1][1] <= height << 1) or (tri[2][0] >= -width and tri[2][0] <= width << 1 and tri[2][1] >= -height and tri[2][1] <= height << 1):
        otri = verticies[tris[i]]

        n = np.asarray(((otri[1,1] - otri[0,1])*(otri[2,2] - otri[0,2]) - (otri[1,2] - otri[0,2])*(otri[2,1] - otri[0,1]),-((otri[1,0] - otri[0,0])*(otri[2,2] - otri[0,2]) - (otri[1,2] - otri[0,2])*(otri[2,0] - otri[0,0])),(otri[1,0] - otri[0,0])*(otri[2,1] - otri[0,1]) - (otri[1,1] - otri[0,1])*(otri[2,0] - otri[0,0])),np.float32)
        n = n / math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])

        camRay = otri[0] - camera
        camRay = camRay / math.sqrt(camRay[0]*camRay[0]+camRay[1]*camRay[1]+camRay[2]*camRay[2])

        if (n[0] * camRay[0] + n[1] * camRay[1] + n[2] * camRay[2]) < 0:
          r = n*2*(n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) - light
          shade = (0.5 * (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) + 0.5) + max((r[0] * -camRay[0] + r[1] * -camRay[1] + r[2] * -camRay[2]),0)**10

          match mat:
            case 0:
              color = shade * np.abs(otri[0] * colscale + 25)
            case 2:
              uv = texcoord[texmap[i]]

          #replace vars?
          #replace argsort with numba?
          ysort = np.argsort(tri[:, 1])

          zstart, zmiddle, zend =  1 / (tri[ysort[0]][2] + 1e-32), 1 / (tri[ysort[1]][2] + 1e-32), 1 / (tri[ysort[2]][2] + 1e-32)
          zslope1 = (zend - zstart) / (tri[ysort[2]][1] - tri[ysort[0]][1] + 1e-32)
          zslope2 = (zmiddle - zstart) / (tri[ysort[1]][1] - tri[ysort[0]][1] + 1e-32)
          zslope3 = (zend - zmiddle) / (tri[ysort[2]][1] - tri[ysort[1]][1] + 1e-32)

          xslope1 = (tri[ysort[2]][0] - tri[ysort[0]][0]) / (tri[ysort[2]][1] - tri[ysort[0]][1] + 1e-32)
          xslope2 = (tri[ysort[1]][0] - tri[ysort[0]][0]) / (tri[ysort[1]][1] - tri[ysort[0]][1] + 1e-32)
          xslope3 = (tri[ysort[2]][0] - tri[ysort[1]][0]) / (tri[ysort[2]][1] - tri[ysort[1]][1] + 1e-32)

          uvslope1 = (uv[ysort[2]] * zend - uv[ysort[0]] * zstart) / (tri[ysort[2]][1] - tri[ysort[0]][1] + 1e-32)
          uvslope2 = (uv[ysort[1]] * zmiddle - uv[ysort[0]] * zstart) / (tri[ysort[1]][1] - tri[ysort[0]][1] + 1e-32)
          uvslope3 = (uv[ysort[2]] * zend - uv[ysort[1]] * zmiddle) / (tri[ysort[2]][1] - tri[ysort[1]][1] + 1e-16)

          for y in range(max(0, tri[ysort[0]][1]), min(height, tri[ysort[2]][1] + 1)):
            yc = y - tri[ysort[0]][1]

            x1 = tri[ysort[0]][0] + xslope1 * yc
            uv1 = uv[ysort[0]] * zstart + yc * uvslope1
            z1 = zstart + zslope1 * yc

            if y < tri[ysort[1]][1]:
              x2 = tri[ysort[0]][0] + xslope2 * yc
              uv2 = uv[ysort[0]] * zstart + yc * uvslope2
              z2 = zstart + zslope2 * yc
            else:
              yc = y - tri[ysort[1]][1]
              x2 = tri[ysort[1]][0] + xslope3 * yc
              uv2 = uv[ysort[1]] * zmiddle + yc * uvslope3
              z2 = zmiddle + zslope3 * yc

            if x1 > x2:
              x1, x2 = x2, x1
              uv1, uv2 = uv2, uv1
              z1, z2 = z2, z1

            xc1, xc2 = max(0, min(int(x1 + 1),width)), max(0, min(int(x2 + 1), width))
            if xc1 != xc2:
              zslope = (z2 - z1) / (x2 - x1 + 1e-32)
              uvslope = (uv2 - uv1) / (x2 - x1 + 1e-32)
              match mat:
                case 2:
                  for x in range(xc1, xc2):
                    z = 1 / (z1 + (x - x1) * zslope + 1e-32)
                    if z < zbuffer[x, y]:
                      zbuffer[x, y] = z
                      uvo = (uv1 + (x - x1) * uvslope) * z
                      if min(uvo) >= 0 and max(uvo) <= 1.1:
                        #frame[x, y] = np.minimum(shade * tex[int(uvo[0] * texsize[0]),int(uvo[1] * texsize[1])],255)
                        #"""
                        #replace minimum with custom numba 3d min?
                        ltx = int(uvo[0] * texsize[0])
                        lty = int(uvo[1] * texsize[1])
                        if ltx > texsize[0] or lty > texsize[1]:
                            frame[x, y] = np.minimum(shade * tex[ltx,lty],255)
                        else:
                          ftx = uvo[0] * texsize[0] - ltx
                          fty = uvo[1] * texsize[1] - lty
                          frame[x, y] = np.minimum(shade * ((1-fty)*(ftx*tex[ltx + 1, lty] + (1-ftx)*tex[ltx, lty]) + fty*(ftx*tex[ltx + 1, lty + 1] + (1-ftx)*tex[ltx, lty + 1])),255)
                        #"""
                        #max(0, 100 - z) / 100
                        #frame[x, y] = tex[int(uvo[0]*texsize[0]), int(uvo[1]*texsize[1])]

                        # fog (20 = near 100 = far)
                        if(z > 20): 
                          per = max(min(z-20, 100),0) / 100
                          frame[x, y] = frame[x, y]*(1-per) + bk*per
                case 0:
                  for x in range(xc1, xc2):
                    z = 1 / (z1 + (x - x1) * zslope + 1e-32)
                    if z < zbuffer[x, y]:
                      zbuffer[x, y] = z
                      frame[x, y] = color
                case 3:
                  for x in range(xc1, xc2):
                    z = 1 / (z1 + (x - x1) * zslope + 1e-32)
                    if z < zbuffer[x, y]:
                      zbuffer[x, y] = z
                      frame[x, y] = max(0, 255 - z * 10)

  """
  @staticmethod
  @njit()
  def project3(verticies, projection, centerx, centery):
    vertlen = len(verticies)
    nverts = np.empty((vertlen, 3), np.float32)
    for i in range(vertlen):
      if verticies[i][2] >= 0:
        nvert = matrixvect((verticies[i][0],verticies[i][1],verticies[i][2], 1.0), projection)
        pvert = nvert[:2]/nvert[3]
      else:
        nverts[i][0] = 100000
        nverts[i][1] = nverts[i][0]
        continue
      nverts[i][1] = int((pvert[1] + 1) * centery)
      nverts[i][0] = int((pvert[0] + 1) * centerx)
      nverts[i][2] = nvert[3]
    return nverts
  """
  
  #"""
  @staticmethod
  @njit()
  def projectf(verticies, projection, centerx, centery, pos, yang, xang):
    vertlen = len(verticies)
    ycos = math.cos(-yang)
    ysin = math.sin(-yang)
    xsin = math.sin(-xang)
    xcos = math.cos(-xang)
    nverts = np.empty((vertlen, 3), np.float32)
    for i in range(vertlen):
      zt = (verticies[i][1] - pos[1]) * xsin + ((verticies[i][2] - pos[2]) * ycos - (verticies[i][0] - pos[0]) * ysin) * xcos
      if zt < 0:
        nverts[i][0] = 100000
        nverts[i][1] = nverts[i][0]
        continue
      nverts[i][1] = int(((((verticies[i][1] - pos[1]) * xcos - ((verticies[i][2] - pos[2]) * ycos - (verticies[i][0] - pos[0]) * ysin) * xsin)*projection[1, 1])/zt + 1) * centery)
      nverts[i][0] = int(((((verticies[i][0] - pos[0]) * ycos + (verticies[i][2] - pos[2]) * ysin)*projection[0, 0])/zt + 1) * centerx)
      nverts[i][2] = zt
    return nverts
  #"""

  @staticmethod
  @njit()
  def _renfil(zbuf, surf, bk, pos, ang, npla):
    zbuf[:,:] = 1e32
    surf[:,:] = bk
    return np.asarray(((pos[0]+(npla[0,0] * math.cos(ang[1]) + (npla[0,1] * math.sin(ang[0]) + npla[0,2] * math.cos(ang[0])) * math.sin(ang[1])), pos[1]+(npla[0,1] * math.cos(ang[0]) - npla[0,2] * math.sin(ang[0])), pos[2]+(npla[0,1] * math.sin(ang[0]) + npla[0,2] * math.cos(ang[0]) * math.cos(ang[1]) - npla[0,0] * math.sin(ang[1]))), (npla[1,0] * math.cos(ang[1]) + (npla[1,1] * math.sin(ang[0]) + npla[1,2] * math.cos(ang[0])) * math.sin(ang[1]), npla[1,1] * math.cos(ang[0]) - npla[1,2] * math.sin(ang[0]), (npla[1,1] * math.sin(ang[0]) + npla[1,2] * math.cos(ang[0])) * math.cos(ang[1]) - npla[1,0] * math.sin(ang[1]))),np.float32)

  def render(self, scene):
    #translateval = np.asarray(list(map(lambda x: -x, self.camera.position)),dtype=np.float32)
    nplane = self._renfil(self.zbuffer, self.surface, scene.background, self.camera.position, self.camera.angle, self.nearplane)
    for object in scene.objects:
      '''
      toverts = self.translate(object[0].verts, object[1])
      #clip these triangles here
      rfaces, overts, rmap, rcoord = self.cliptri(self.nearplane,object[0].mat, self.uv,object[0].faces,self.rotate(self.translate(toverts, translateval), -self.camera.yang, -self.camera.xang),object[0].texcoord,object[0].texmap)

      rverts = self.proj(overts)

      rlight = Light(self.rotate(np.asarray([scene.light.dir], np.float32), -self.camera.yang, -self.camera.xang)[0].astype(np.float32))

      if object[0].mat == 1:
        rmat = 2
      else:
        rmat = object[0].mat
      '''

      rfaces, overts, rmap, rcoord = self.cliptri(nplane, object.mat, self.uv, object.faces,object.verts, object.texcoord, object.texmap, object.flen, object.vlen, object.clen)

      match len(rfaces):
        case 0:
          continue

      rmat = object.mat
      match rmat:
        case 1:
          rmat = 2

      """
      rverts = self.proj(self.rotate(self.translate(overts, translateval), -self.camera.yang,-self.camera.xang))
      rmap = object[0].texmap
      rcoord = object[0].texcoord
      """

      self.rendertris(scene.background, overts, self.projectf(overts,self.projection, self.centerx,self.centery,self.camera.position,self.camera.angle[1],self.camera.angle[0]), rfaces, self.camera.position, scene.light.dir, self.width, self.height, self.surface, self.zbuffer, rmat, object.tex, object.texsize, rmap, rcoord)

  @staticmethod
  @njit()
  def translate(verticies, position):
    vertlen = len(verticies)
    nverts = np.empty((vertlen, 3), np.float32)
    for i in range(vertlen):
      nverts[i] = verticies[i][0] + position[0], verticies[i][1] + position[1], verticies[i][2] + position[2]
    return nverts

  @staticmethod
  @njit()
  def rotate(verticies, yang, xang):
    vertlen = len(verticies)
    nverts = np.empty((vertlen, 3), np.float32)
    ycos = math.cos(yang)
    ysin = math.sin(yang)
    xsin = math.sin(xang)
    xcos = math.cos(xang)
    for i in range(vertlen):
      nverts[i] = verticies[i][0] * ycos + verticies[i][2] * ysin, verticies[i][1] * xcos - (verticies[i][2] * ycos - verticies[i][0] * ysin) * xsin,verticies[i][1] * xsin + (verticies[i][2] * ycos - verticies[i][0] * ysin) * xcos
    return nverts

  @staticmethod
  @njit()
  def rotatem(vert, yang, xang):
    return np.asarray((vert[0] * math.cos(yang) + (vert[1] * math.sin(xang) + vert[2] * math.cos(xang)) * math.sin(yang), vert[1] * math.cos(xang) - vert[2] * math.sin(xang), (vert[1] * math.sin(xang) + vert[2] * math.cos(xang)) * math.cos(yang) - vert[0] * math.sin(yang)),np.float32)

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
    if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]: elapsed_time *= 5
    if keys[pg.K_LCTRL]: elapsed_time *= 0.25
    if keys[pg.K_w]:
      self.camera.position[0] += elapsed_time * math.sin(self.camera.yang)
      self.camera.position[2] += elapsed_time * math.cos(self.camera.yang)
    if keys[pg.K_s]:
      self.camera.position[0] -= elapsed_time * math.sin(self.camera.yang)
      self.camera.position[2] -= elapsed_time * math.cos(self.camera.yang)
    if keys[pg.K_a]:
      self.camera.position[0] -= elapsed_time * math.cos(self.camera.yang)
      self.camera.position[2] += elapsed_time * math.sin(self.camera.yang)
    if keys[pg.K_d]:
      self.camera.position[0] += elapsed_time * math.cos(self.camera.yang)
      self.camera.position[2] -= elapsed_time * math.sin(self.camera.yang)
    if keys[pg.K_e]: self.camera.position[1] += elapsed_time
    if keys[pg.K_q]: self.camera.position[1] -= elapsed_time
    if keys[pg.K_LEFT]: self.camera.yang -= elapsed_time / 2
    if keys[pg.K_RIGHT]: self.camera.yang += elapsed_time / 2
    if keys[pg.K_UP]: self.camera.xang -= elapsed_time / 2
    if keys[pg.K_DOWN]: self.camera.xang += elapsed_time / 2

  @staticmethod
  @njit()
  def smmovecalc(kw,ks,ka,kd,ke,kq,kl,kr,ku,kdo, velocity, elapsed_time, position, angle, speed):
    #force is 2 but friction is 1

    xf = 0
    yf = 0
    zf = 0

    if kw and velocity[0] < speed: xf = 20
    if ks and velocity[0] > -speed: xf = -20
    if ka and velocity[1] > -speed: yf = -20
    if kd and velocity[1] < speed : yf = 20
    if ke and velocity[2] < speed : zf = 20
    if kq and velocity[2] > -speed : zf = -20

    if kw or ks:
      velocity[0] += xf*elapsed_time
    else:
      if velocity[0] > 0: 
        xf -= 20
        velocity[0] += xf*elapsed_time
        if velocity[0] < 0: velocity[0] = 0
      elif velocity[0] < 0: 
        xf += 20
        velocity[0] += xf*elapsed_time
        if velocity[0] > 0: velocity[0] = 0
    if ka or kd:
      velocity[1] += yf*elapsed_time
    else:
      if velocity[1] > 0: 
        yf -= 20
        velocity[1] += yf*elapsed_time
        if velocity[1] < 0:
          velocity[1] = 0
      elif velocity[1] < 0: 
        yf += 20
        velocity[1] += yf*elapsed_time
        if velocity[1] > 0: velocity[1] = 0
    if ke or kq:
      velocity[2] += zf*elapsed_time
    else:
      if velocity[2] > 0: 
        zf -= 20
        velocity[2] += zf*elapsed_time
        if velocity[2] < 0: velocity[2] = 0
      elif velocity[2] < 0: 
        zf += 20
        velocity[2] += zf*elapsed_time
        if velocity[2] > 0: velocity[2] = 0

    position[0] += (velocity[1]*elapsed_time) * math.cos(angle[1]) + (velocity[0]*elapsed_time) * math.sin(angle[1]) 
    position[2] -= (velocity[1]*elapsed_time) * math.sin(angle[1]) - (velocity[0]*elapsed_time) * math.cos(angle[1])
    position[1] += (velocity[2]*elapsed_time)

    if kl: angle[1] -= elapsed_time / 1.5
    if kr: angle[1] += elapsed_time / 1.5
    if ku: angle[0] -= elapsed_time / 1.5
    if kdo: angle[0] += elapsed_time / 1.5

  def move(self, elapsed_time):
    keys = pg.key.get_pressed()
    self.smmovecalc(keys[119], keys[115], keys[97], keys[100], keys[101], keys[113], keys[1073741904], keys[1073741903], keys[1073741906], keys[1073741905],self.camera.velocity, elapsed_time, self.camera.position, self.camera.angle, self.camera.speed)

  #handle faces and rverts and remember to use np.where too
  @staticmethod
  @njit()
  def cliptri(plane, mat, uv, faces, verts, coord, texmap, flen, vind, cind):
    plane[1] = plane[1] / math.sqrt(plane[1,0]*plane[1,0]+plane[1,1]*plane[1,1]+plane[1,2]*plane[1,2])
    find = 0
    nfaces = np.empty((flen << 1, 3), np.uint32)
    match mat:
      case 1:
        coord = uv
        texmap[:,0] = 0
        texmap[:,1] = 1
        texmap[:,2] = 2
        mat = 2

    #arraylist things
    nmap = np.empty((flen << 1, 3), np.uint32)
    #only store new verticies?
    verts = np.resize(verts, (vind*3, 3))
    ncoord = np.resize(coord, (cind+(vind << 1), 2))

    for i in range(flen):
      face = faces[i]
      tri = np.asarray(((verts[face[0]][0],verts[face[0]][1],verts[face[0]][2]), (verts[face[1]][0],verts[face[1]][1],verts[face[1]][2]), (verts[face[2]][0],verts[face[2]][1],verts[face[2]][2])), np.float32)
      voutn = 0
      vinn = 0
      vout = np.empty(3, np.uint32)
      vin = np.empty(3, np.uint32)
      for j in range(3):
        v = tri[j]
        if (v[0] * plane[1][0] + v[1] * plane[1][1] + v[2] * plane[1][2]) - (plane[0][0] * plane[1][0] + plane[0][1] * plane[1][1] + plane[0][2] * plane[1][2]) < 0:
          vout[voutn] = j
          voutn += 1
        else:
          vin[vinn] = j
          vinn += 1
      match voutn:
        case 0:
          nfaces[find] = face
          match mat:
            case 2:
              nmap[find] = texmap[i]
          find += 1
          continue
        case 1:
          verts[vind], t = intersect(tri[vout[0]], tri[vin[0]], plane[1], plane[0])
          nfaces[find,vin[0]] = face[vin[0]]
          nfaces[find,vin[1]] = face[vin[1]]
          nfaces[find,vout[0]] = vind
          vind += 1
          match mat:
            case 2:
              coord[cind] = t * (coord[texmap[i, vin[0]]] - coord[texmap[i, vout[0]]]) + coord[texmap[i, vin[0]]]
              nmap[find,vin[0]] = texmap[i, vin[0]]
              nmap[find,vin[1]] = texmap[i, vin[1]]
              nmap[find,vout[0]] = cind
              verts[vind], t = intersect(tri[vout[0]], tri[vin[1]], plane[1], plane[0])
              find += 1
              nfaces[find,vout[0]] = vind-1
              nfaces[find,vin[0]] = face[vin[1]]
              nfaces[find,vin[1]] = vind
              nmap[find,vout[0]] = cind
              vind += 1
              cind += 1
              coord[cind] = t * (coord[texmap[i, vin[1]]] - coord[texmap[i, vout[0]]]) + coord[texmap[i, vin[1]]]
              nmap[find,vin[0]] = texmap[i, vin[1]]
              nmap[find,vin[1]] = cind
              cind += 1
              find += 1
            case _:
              verts[vind], t = intersect(tri[vout[0]], tri[vin[1]], plane[1], plane[0])
              find += 1
              nfaces[find,vout[0]] = vind-1
              nfaces[find,vin[0]] = face[vin[1]]
              nfaces[find,vin[1]] = vind
              find += 1
              vind += 1

          continue
        case 2:
          verts[vind], t = intersect(tri[vout[0]], tri[vin[0]], plane[1], plane[0])
          nfaces[find,vout[0]] = vind
          vind += 1
          match mat:
            case 2:
              coord[cind] = t * (coord[texmap[i, vin[0]]] - coord[texmap[i, vout[0]]]) + coord[texmap[i, vin[0]]] 
              nmap[find,vout[0]] = cind
              verts[vind], t = intersect(tri[vout[1]], tri[vin[0]], plane[1], plane[0])
              cind += 1
              coord[cind] = t * (coord[texmap[i, vin[0]]] - coord[texmap[i, vout[1]]]) + coord[texmap[i, vin[0]]]
              nmap[find,vin[0]] = texmap[i, vin[0]]
              nmap[find,vout[1]] = cind
              cind += 1
            case _:
              verts[vind], t = intersect(tri[vout[1]], tri[vin[0]], plane[1], plane[0])

          nfaces[find,vin[0]] = face[vin[0]]
          nfaces[find,vout[1]] = vind
          find += 1
          vind += 1
          continue
        case 3:
          continue

    return nfaces[:find], verts[:vind], nmap[:find], coord[:cind]


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
def intersect(vt, vp, pn, pp):
  vn = vp - vt
  t = ((pp[0] - vp[0])*pn[0] + (pp[1] - vp[1])*pn[1] + (pp[2] - vp[2])*pn[2]) / ((vn[0] * pn[0] + vn[1] * pn[1] + vn[2] * pn[2]) + 1e-32)
  return np.asarray((vn[0] * t + vp[0], vn[1] * t + vp[1], vn[2] * t + vp[2]),np.float32), t


@njit()
def magn(v):
  return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


@njit()
def norm(v):
  return v / math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

@njit()
def cross(a,b):
  return np.asarray((a[1]*b[2] - a[2]*b[1],-(a[0]*b[2] - a[2]*b[0]),a[0]*b[1] - a[1]*b[0]),np.float32)