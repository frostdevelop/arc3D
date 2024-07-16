import numpy as np
import pygame as pg
from numba import njit, jit, int32, float32, float64
from numba.experimental import jitclass
from numba.typed import List


#@jitclass([('verts', float32[:, :]), ("faces", uint16[:, :])])
class Object:

  def __init__(self, verts, faces, mat, **kwargs):
    self.verts = np.asarray(verts, np.float32)
    self.faces = np.asarray(faces, np.uint16)
    self.mat = mat
    if mat == 1 or mat == 2:
      self.tex = pg.surfarray.array3d(pg.image.load(kwargs.get(
          "tex", ""))).astype(np.uint8)
      self.texsize = np.array([len(self.tex) - 1,len(self.tex[0]) - 1]).astype(np.uint16)
    else:
      self.tex = np.empty((1, 3), dtype=np.uint8)
      self.texsize = np.empty(2, dtype=np.uint16)
    if mat == 2:
      self.texmap = np.array(kwargs.get("texmap",  np.empty(1))).astype(np.uint32)
      self.texcoord = np.array(kwargs.get("texcoord",         np.empty(1))).astype(np.float32)
    else:
      self.texmap = np.zeros((len(faces), 3)).astype(np.uint32)
      self.texcoord = np.zeros((1, 2)).astype(np.float32)


@jitclass([("position", float32[:]), ("xang", float64), ("yang", float64),
           ("vfov", float32), ("hfov", float32), ("zfar", float32),
           ("znear", float32)])
class Camera:

  def __init__(self, vfov, position, yang, xang, zfar, znear):
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
    self.nearplane = np.asarray(((0, 0, camera.znear), (0, 0, 1)), np.float32)
    self.zbuffer = np.ones((width, height), dtype=np.float32)
    self.projection = np.empty((4, 4), np.float32)
    self.projection[0, 0] = 1 / (np.tan(self.camera.vfov / 2) *
                                 (self.width / self.height))
    self.projection[1, 1] = 1 / np.tan(self.camera.vfov / 2)
    self.projection[2, 2] = self.camera.zfar / (self.camera.zfar -
                                                self.camera.znear)
    self.projection[2, 3] = 1
    self.projection[3, 2] = (-self.camera.zfar * self.camera.znear) / (
        self.camera.zfar - self.camera.znear)
    self.uv = np.asarray(uv, dtype=np.float32)
    self.mouse = mouse
    self.velocity = np.zeros(3,np.float32)
    self.speed = 2
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
                 iuv=np.empty((3, 2), dtype=np.float32),
                 texmap=np.empty((1, 3), dtype=np.uint32),
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

      n = norm(np.cross(vet1, vet2).astype(np.float32))

      camRay = (verticies[trii[0]] - camera) / rverts[trii[0]][3]

      trix = (rverts[trii[0]][0], rverts[trii[1]][0], rverts[trii[2]][0])
      triy = (rverts[trii[0]][1], rverts[trii[1]][1], rverts[trii[2]][1])
      zmin = min((verticies[trii[0]][3], verticies[trii[1]][3], verticies[trii[2]][3]))

      if dot3d(n, camRay) < 0 and (
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

        zstart, zmiddle, zend = tri[ysort[0]][3], tri[ysort[1]][3], tri[
            ysort[2]][3]

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

        #clip here than reuse uvslope (just make new values for y and x range)

        for y in range(max(0, ystart), min(height, yend + 1)):
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

          xc1, xc2 = max(0, min(int(x1 + 1),width)), max(0, min(int(x2 + 1), width))
          if xc1 != xc2:
            zslope = (z2 - z1) / (x2 - x1 + 1e-32)
            uvslope = (uv2 - uv1) / (x2 - x1 + 1e-32)
            for x in range(xc1, xc2):
              cx = x - x1
              z = 1 / (z1 + cx * zslope + 1e-32)
              if z < zbuffer[x, y]:
                zbuffer[x, y] = z
                if mat == 1 or mat == 2:
                  uvo = (uv1 + cx * uvslope) * z
                  if min(uvo) >= 0 and max(uvo) <= 1.1:
                    frame[x, y] = shade[i] * tex[int(uvo[0] * texsize[0]),int(uvo[1] * texsize[1])]
                    #max(0, 100 - z) / 100
                    #frame[x, y] = tex[int(uvo[0]*texsize[0]), int(uvo[1]*texsize[1])]
                  else:
                    frame[x, y] = (255,0,0)
                elif mat == 0:
                  frame[x, y] = color
                elif mat == 3:
                  frame[x, y] = max(0, 255 - z * 10)

                #fog
                if z > 50: frame[x, y] = frame[x, y]+(bk-frame[x, y])*(min(z-50, 100) / 100)

  @staticmethod
  @njit()
  def project3(verticies, projection, centerx, centery):
    nverts = np.empty((len(verticies), 4), dtype=np.float32)
    for i in range(len(verticies)):
      nvert = matrixvect((verticies[i][0],verticies[i][1],verticies[i][2], 1.0), projection)
      if nvert[3] >= 0:
        nvert = np.append(np.divide(nvert[:3], nvert[3]), nvert[3] + 1e-16).astype(np.float32)
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
    #translateval = np.asarray(list(map(lambda x: -x, self.camera.position)),dtype=np.float32)
    translateval = np.asarray((-self.camera.position[0],-self.camera.position[1],-self.camera.position[2]),dtype=np.float32)
    self.zbuffer[:, :] = 1e32
    self.surface[:, :] = scene.background
    nplane = np.asarray((self.camera.position+self.rotatem(self.nearplane[0], self.camera.yang,self.camera.xang),self.rotatem(self.nearplane[1],self.camera.yang,self.camera.xang)),np.float32)
    for object in scene.objects:

      '''
      if object[0].mat != 1:
        continue
      '''

      #print(object[0].mat)

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

      """
      rfaces, overts, rmap, rcoord = self.cliptri(nplane, object[0].mat, self.uv, object[0].faces,self.translate(object[0].verts, object[1]), object[0].texcoord, object[0].texmap)
      if object[0].mat == 1:
        rmat = 2
      else:
        rmat = object[0].mat
      rverts = self.project3(self.rotate(self.translate(overts, translateval), -self.camera.yang,-self.camera.xang),self.projection, self.centerx,self.centery)
      """

      rfaces, overts, rmap, rcoord = self.cliptri(nplane, object[0].mat, self.uv, object[0].faces,self.translate(object[0].verts, object[1]), object[0].texcoord, object[0].texmap)
      rmat = object[0].mat
      if rmat == 1:
        rmat = 2
      rverts = self.project3(self.rotate(self.translate(overts, translateval), -self.camera.yang,-self.camera.xang),self.projection, self.centerx,self.centery)

      """
      rverts = self.proj(self.rotate(self.translate(overts, translateval), -self.camera.yang,-self.camera.xang))
      rmap = object[0].texmap
      rcoord = object[0].texcoord
      """
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
      '''
      print(rfaces)
      print(rmap)
      print(rcoord)
      print(rverts)
      '''

      self.rendertris(scene.background, overts, rverts, rfaces, self.camera.position, scene.light, self.width, self.height, self.surface, self.zbuffer, rmat, object[0].tex, object[0].texsize, self.uv, rmap, rcoord)

  @staticmethod
  @njit()
  def translate(verticies, position):
    nverts = np.empty(shape=(len(verticies), 3))
    for i in range(len(verticies)):
      vert = verticies[i]
      nverts[i] = (
          vert[0] + position[0], vert[1] + position[1], vert[2] + position[2]
      )
    return nverts

  @staticmethod
  @njit()
  def rotate(verticies, yang, xang):
    nverts = np.empty((len(verticies), 3), np.float32)
    ycos = np.cos(yang)
    ysin = np.sin(yang)
    xsin = np.sin(xang)
    xcos = np.cos(xang)
    for i in range(len(verticies)):
      vert = verticies[i]
      tv = vert[2] * ycos - vert[0] * ysin
      nverts[i] = (
        vert[0] * ycos + vert[2] * ysin, vert[1] * xcos - tv * xsin,
        vert[1] * xsin + tv * xcos
      )
    return nverts

  @staticmethod
  @njit()
  def rotatem(vert, yang, xang):
    return np.asarray((vert[0] * np.cos(yang) + vert[2] * np.sin(yang), vert[1] * np.cos(xang) - vert[2] * np.sin(xang),(vert[2] * np.cos(yang) - vert[0] * np.sin(yang))*(vert[1] * np.sin(xang) + vert[2] * np.cos(xang))),np.float32)

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

  def smoothmove(self, elapsed_time):
    keys = pg.key.get_pressed()

    add = elapsed_time*self.speed

    if keys[pg.K_w]: self.velocity[0] += add
    if keys[pg.K_s]: self.velocity[0] -= add
    if keys[pg.K_a]: self.velocity[1] -= add
    if keys[pg.K_d]: self.velocity[1] += add

    if self.velocity[0] > self.speed:
      self.velocity[0] = self.speed
    elif self.velocity[0] < -self.speed:
      self.velocity[0] = -self.speed

    if self.velocity[1] > self.speed:
      self.velocity[1] = self.speed
    elif self.velocity[1] < -self.speed:
      self.velocity[1] = -self.speed

    self.camera.position[0] += self.velocity[0] * np.sin(self.camera.yang) 
    self.camera.position[2] += self.velocity[0] * np.cos(self.camera.yang)
    self.camera.position[0] += self.velocity[1] * np.cos(self.camera.yang)
    self.camera.position[2] -= self.velocity[1] * np.sin(self.camera.yang)


    if keys[pg.K_e]: self.velocity[2] += add
    if keys[pg.K_q]: self.velocity[2] -= add

    self.camera.position[1] += self.velocity[2]

    if keys[pg.K_LEFT]: self.camera.yang -= elapsed_time / 2
    if keys[pg.K_RIGHT]: self.camera.yang += elapsed_time / 2
    if keys[pg.K_UP]: self.camera.xang -= elapsed_time / 2
    if keys[pg.K_DOWN]: self.camera.xang += elapsed_time / 2

    #print(elapsed_time)

    mult = 0.1/elapsed_time
    if mult > 0.9: mult = 0.9
    #print(mult)
    #print(self.velocity)

    self.velocity[0] *= mult
    self.velocity[1] *= mult
    self.velocity[2] *= mult


  """
  @staticmethod
  @njit()
  def cliptris(verts,faces,coord,map,plane):
    ntris = np.empty((0,3), dtype=np.float32)
    nmap = np.empty((0,3), dtype=np.uint16)
    for i in range(len(tris)):
      ntriv,nmapv,coord = cliptri(tris[i,0],coord,map[i,0],plane)
      ntris = np.concatnate(ntris, ntriv)
      nmap = np.concatnate(nmap, nmapv)
    return ntris,nmap,coord
    """

  #handle faces and rverts and remember to use np.where too
  @staticmethod
  @njit()
  def cliptri(plane, mat, uv, faces, verts, coord, texmap):
    plane[1] = norm(plane[1])
    nfaces = np.empty((0, 3), dtype=np.uint32)
    if mat == 1:
        coord = uv
        texmap[:] = np.asarray((0,1,2),np.uint32)
    nmap = np.empty((0, 3), dtype=np.uint32)
    for i in range(len(faces)):
      face = faces[i]
      tri = np.asarray(((verts[face[0]][0],verts[face[0]][1],verts[face[0]][2]), (verts[face[1]][0],verts[face[1]][1],verts[face[1]][2]), (verts[face[2]][0],verts[face[2]][1],verts[face[2]][2])), dtype=np.float32)
      lnpnd = lambda v, plane: dot3d(v, plane[1]) - dot3d(plane[0], plane[1])
      voutn = 0
      vinn = 0
      vout = np.empty(3, dtype=np.uint32)
      vin = np.empty(3, dtype=np.uint32)
      for j in range(3):
        v = tri[j]
        if lnpnd(v, plane) < 0:
          vout[voutn] = j
          voutn += 1
        else:
          vin[vinn] = j
          vinn += 1
      match voutn:
        case 0:
          nfaces = np.append(nfaces, np.asarray([(face[0],face[1],face[2])],np.uint32), axis=0)
          if mat == 2 or mat == 1:
            nmap = np.append(nmap, np.asarray([(texmap[i][0],texmap[i][1],texmap[i][2])],np.uint32), axis=0)
          continue
        case 1:
          nt = np.empty(3, dtype=np.uint32)
          nt[vin[0]] = face[vin[0]]
          nt[vin[1]] = face[vin[1]]
          nv, t = intersect(tri[vout[0]], tri[vin[0]], plane[1], plane[0])
          verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
          nt[vout[0]] = len(verts) - 1
          nfaces = np.append(nfaces, np.asarray([(nt[0],nt[1],nt[2])],np.uint32), axis=0)
          if mat == 2 or mat == 1:
            nm = np.empty(3, dtype=np.uint32)
            nm[vin[0]] = texmap[i, vin[0]]
            nm[vin[1]] = texmap[i, vin[1]]
            ntex = t * (coord[texmap[i, vin[0]]] - coord[texmap[i, vout[0]]]) + coord[texmap[i, vin[0]]]
            coord = np.append(coord,np.asarray([(ntex[0],ntex[1])],np.float32),axis=0)
            nm[vout[0]] = len(coord) - 1
            nmap = np.append(nmap, np.asarray([(nm[0],nm[1],nm[2])]), axis=0)
            nvm = np.empty(3, dtype=np.uint32)
            nvm[vout[0]] = nm[vout[0]]
            nvm[vin[0]] = texmap[i, vin[1]]
            nvt = np.empty(3, dtype=np.uint32)
            nvt[vout[0]] = len(verts) - 1
            nvt[vin[0]] = face[vin[1]]
            nv, t = intersect(tri[vout[0]], tri[vin[1]], plane[1], plane[0])
            verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
            nvt[vin[1]] = len(verts) - 1
            nfaces = np.append(nfaces, np.asarray([(nvt[0],nvt[1],nvt[2])],np.uint32), axis=0)
            ntex = t * (coord[texmap[i, vin[1]]] - coord[texmap[i, vout[0]]]) + coord[texmap[i, vin[1]]]
            coord = np.append(coord, np.asarray([(ntex[0],ntex[1])],np.float32),axis=0)
            nvm[vin[1]] = len(coord) - 1
            nmap = np.append(nmap, np.asarray([(nvm[0],nvm[1],nvm[2])],np.uint32), axis=0)
          else:
            nvt = np.empty(3, dtype=np.uint32)
            nvt[vout[0]] = len(verts) - 1
            nvt[vin[0]] = face[vin[1]]
            nv, t = intersect(tri[vout[0]], tri[vin[1]], plane[1], plane[0])
            verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
            nvt[vin[1]] = len(verts) - 1
            nfaces = np.append(nfaces, np.asarray([(nvt[0],nvt[1],nvt[2])],np.uint32), axis=0)
          #print(nvt)
          #print(verts)
          #print("onedetected")

          continue
        case 2:
          nt = np.empty(3, dtype=np.uint32)
          nt[vin[0]] = face[vin[0]]
          nv, t = intersect(tri[vout[0]], tri[vin[0]], plane[1], plane[0])
          verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
          nt[vout[0]] = len(verts)-1
          if mat == 2 or mat == 1:
            nm = np.empty(3, dtype=np.uint32)
            nm[vin[0]] = texmap[i, vin[0]]
            ntex = t * (coord[texmap[i, vin[0]]] - coord[texmap[i, vout[0]]]) + coord[texmap[i, vin[0]]]
            coord = np.append(coord, np.asarray([(ntex[0],ntex[1])],np.float32),axis=0)
            nm[vout[0]] = len(coord) - 1
            nv, t = intersect(tri[vout[1]], tri[vin[0]], plane[1], plane[0])
            verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
            nt[vout[1]] = len(verts)-1
            ntex = t * (coord[texmap[i, vin[0]]] - coord[texmap[i, vout[1]]]) + coord[texmap[i, vin[0]]]
            coord = np.append(coord, np.asarray([(ntex[0],ntex[1])],np.float32),axis=0)
            nm[vout[1]] = len(coord) - 1
            nmap = np.append(nmap, np.asarray([(nm[0],nm[1],nm[2])],np.uint32), axis=0)
          else:
            nv, t = intersect(tri[vout[1]], tri[vin[0]], plane[1], plane[0])
            verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
            nt[vout[1]] = len(verts)-1
          nfaces = np.append(nfaces, np.asarray([(nt[0],nt[1],nt[2])],np.uint32), axis=0)
          continue
        case 3:
          continue
    return nfaces, verts, nmap, coord


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
  nd = dot3d(vn, pn)
  pd = dot3d(pp - vp, pn)
  if nd == 0: nd = 1e-16
  if pd == 0: pd = 1e-16
  t = pd / nd
  pos = np.asarray((vn[0] * t + vp[0], vn[1] * t + vp[1], vn[2] * t + vp[2]),np.float32)
  return pos, t


#print(intersect(np.asarray([5.1,3.5,-9.2]),np.asarray([-4.9,-6.5,-5.3]),np.asarray([5,5,-4.9]),np.asarray([-1.71,20,-7.8])))
#to do clipping


@njit()
def magn(v):
  return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


@njit()
def norm(v):
  return v / magn(v)
