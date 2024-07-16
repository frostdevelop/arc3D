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
      case _:
        self.texmap = np.zeros((len(faces), 3)).astype(np.uint32)
        self.texcoord = np.zeros((1, 2)).astype(np.float32)

  def upd(self):
    self.verts = Renderer.rotate(self.mverts, self.rotation[0], self.rotation[1])
    self.verts = Renderer.translate(self.verts, self.position)

@jitclass([("position", float32[:]), ("angle", float32[:]),
           ("vfov", float32), ("hfov", float32), ("zfar", float32),
           ("znear", float32), ("velocity", float32[:]), ("speed", float32)])
class Camera:

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
    self.centerx = width >> 1
    self.centery = height >> 1
    self.surface = np.ones((width, height, 3), dtype=np.uint8)
    self.nearplane = np.asarray(((0, 0, camera.znear), (0, 0, 1)), np.float32)
    self.zbuffer = np.empty((width, height), dtype=np.float32)
    self.projection = np.zeros((4, 4), np.float32)
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
    #just pass neccessary fields from camera
    match mat:
      case 0:
        colscale = 230 / np.max(np.abs(verticies))

    uv = np.empty((3, 2), dtype=np.float32)
    for i in range(len(tris)):
      trii = tris[i]
      tri = rverts[trii]
      otri = verticies[trii]

      n = norm(np.cross(otri[1] - otri[0], otri[2] - otri[0]).astype(np.float32))

      camRay = (otri[0] - camera) / tri[0][2]
      #zmin = min(rverts[trii[0]][3], rverts[trii[1]][3], rverts[trii[2]][3])

      if dot3d(n, camRay) < 0 and (
          (tri[0][0] >= -width and tri[0][0] <= width << 1 and tri[0][1] >= -height
           and tri[0][1] <= height << 1) or
          (tri[1][0] >= -width and tri[1][0] <= width << 1 and tri[1][1] >= -height
           and tri[1][1] <= height << 1) or
          (tri[2][0] >= -width and tri[2][0] <= width << 1 and tri[2][1] >= -height
           and tri[2][1] <= height << 1)):

        shade = min(1, max(0, 0.5 * dot3d(n, light.dir) + 0.6))

        match mat:
          case 0:
            color = shade * np.abs(otri[0] * colscale + 25)
          case 2:
            uv = texcoord[texmap[i]]


        #if tri[:,0].min() < 0 or tri[:,0].max() > width or tri[:,1].min() < 0 or tri[:,1].max() > height: continue

        ysort = np.argsort(tri[:, 1])
        xstart, ystart = tri[ysort[0]][:2]
        xmiddle, ymiddle = tri[ysort[1]][:2]
        xend, yend = tri[ysort[2]][:2]

        zstart, zmiddle, zend =  1 / (tri[ysort[0]][2] + 1e-32), 1 / (tri[ysort[1]][2] + 1e-32), 1 / (tri[ysort[2]][2] + 1e-32)
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
            match mat:
              case 2:
                for x in range(xc1, xc2):
                  cx = x - x1
                  z = 1 / (z1 + cx * zslope + 1e-32)
                  if z < zbuffer[x, y]:
                    zbuffer[x, y] = z
                    uvo = (uv1 + cx * uvslope) * z
                    if min(uvo) >= 0 and max(uvo) <= 1.1:
                      frame[x, y] = shade * tex[int(uvo[0] * texsize[0]),int(uvo[1] * texsize[1])]
                      #max(0, 100 - z) / 100
                      #frame[x, y] = tex[int(uvo[0]*texsize[0]), int(uvo[1]*texsize[1])]

                    # fog (20 = near 100 = far)
                    if(z > 20): 
                      per = max(min(z-20, 100),0) / 100
                      frame[x, y] = frame[x, y]*(1-per) + bk*per
              case 0:
                for x in range(xc1, xc2):
                  cx = x - x1
                  z = 1 / (z1 + cx * zslope + 1e-32)
                  if z < zbuffer[x, y]:
                    zbuffer[x, y] = z
                    frame[x, y] = color
              case 3:
                for x in range(xc1, xc2):
                  cx = x - x1
                  z = 1 / (z1 + cx * zslope + 1e-32)
                  if z < zbuffer[x, y]:
                    zbuffer[x, y] = z
                    frame[x, y] = max(0, 255 - z * 10)

  @staticmethod
  @njit()
  def project3(verticies, projection, centerx, centery):
    vertlen = len(verticies)
    nverts = np.empty((vertlen, 3), dtype=np.float32)
    for i in range(vertlen):
      nvert = matrixvect((verticies[i][0],verticies[i][1],verticies[i][2], 1.0), projection)
      if nvert[3] > 0:
        pvert = nvert[:2]/nvert[3]
      else:
        pvert = nvert[:2]
      nverts[i][1] = int((-pvert[1] + 1) * centery)
      nverts[i][0] = int((pvert[0] + 1) * centerx)
      nverts[i][2] = nvert[3]
    return nverts

  @staticmethod
  @njit()
  def _renfil(zbuf, surf, bk):
    zbuf[:,:] = 1e32
    surf[:,:] = bk

  def render(self, scene):
    #translateval = np.asarray(list(map(lambda x: -x, self.camera.position)),dtype=np.float32)
    self._renfil(self.zbuffer, self.surface, scene.background)
    nplane = np.asarray((self.camera.position+self.rotatem(self.nearplane[0], self.camera.angle[1],self.camera.angle[0]),self.rotatem(self.nearplane[1],self.camera.angle[1],self.camera.angle[0])),np.float32)
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

      rfaces, overts, rmap, rcoord = self.cliptri(nplane, object.mat, self.uv, object.faces,object.verts, object.texcoord, object.texmap)

      match len(rfaces):
        case 0:
          continue

      rmat = object.mat
      match rmat:
        case 1:
          rmat = 2

      #pass length as argument?
      rverts = self.project3(self.rotate(self.translate(overts, np.asarray((-self.camera.position[0],-self.camera.position[1],-self.camera.position[2]), np.float32)), -self.camera.angle[1],-self.camera.angle[0]),self.projection, self.centerx,self.centery)

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

      self.rendertris(scene.background, overts, rverts, rfaces, self.camera.position, scene.light, self.width, self.height, self.surface, self.zbuffer, rmat, object.tex, object.texsize, rmap, rcoord)

  @staticmethod
  @njit()
  def translate(verticies, position):
    vertlen = len(verticies)
    nverts = np.empty((vertlen, 3), np.float32)
    for i in range(vertlen):
      vert = verticies[i]
      nverts[i] = (
          vert[0] + position[0], vert[1] + position[1], vert[2] + position[2]
      )
    return nverts

  @staticmethod
  @njit()
  def rotate(verticies, yang, xang):
    #check if putting in list vs sep ones
    vertlen = len(verticies)
    nverts = np.empty((vertlen, 3), np.float32)
    ycos = np.cos(yang)
    ysin = np.sin(yang)
    xsin = np.sin(xang)
    xcos = np.cos(xang)
    for i in range(vertlen):
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

  #"""
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

    if not (kw or ks):
      if velocity[0] > 0: 
        xf -= 20
        velocity[0] += xf*elapsed_time
        if velocity[0] < 0: velocity[0] = 0
      elif velocity[0] < 0: 
        xf += 20
        velocity[0] += xf*elapsed_time
        if velocity[0] > 0: velocity[0] = 0
    else:
      velocity[0] += xf*elapsed_time
    if not (ka or kd):
      if velocity[1] > 0: 
        yf -= 20
        velocity[1] += yf*elapsed_time
        if velocity[1] < 0:
          velocity[1] = 0
      elif velocity[1] < 0: 
        yf += 20
        velocity[1] += yf*elapsed_time
        if velocity[1] > 0: velocity[1] = 0
    else:
      velocity[1] += yf*elapsed_time
    if not (ke or kq):
      if velocity[2] > 0: 
        zf -= 20
        velocity[2] += zf*elapsed_time
        if velocity[2] < 0: velocity[2] = 0
      elif velocity[2] < 0: 
        zf += 20
        velocity[2] += zf*elapsed_time
        if velocity[2] > 0: velocity[2] = 0
    else:
      velocity[2] += zf*elapsed_time

    position[0] += (velocity[1]*elapsed_time) * np.cos(angle[1]) + (velocity[0]*elapsed_time) * np.sin(angle[1]) 
    position[2] -= (velocity[1]*elapsed_time) * np.sin(angle[1]) - (velocity[0]*elapsed_time) * np.cos(angle[1])
    position[1] += (velocity[2]*elapsed_time)

    if kl: angle[1] -= elapsed_time / 1.5
    if kr: angle[1] += elapsed_time / 1.5
    if ku: angle[0] -= elapsed_time / 1.5
    if kdo: angle[0] += elapsed_time / 1.5
  #"""

  def move(self, elapsed_time):
    keys = pg.key.get_pressed()
    self.smmovecalc(keys[119], keys[115], keys[97], keys[100], keys[101], keys[113], keys[1073741904], keys[1073741903], keys[1073741906], keys[1073741905],self.camera.velocity, elapsed_time, self.camera.position, self.camera.angle, self.camera.speed)

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
    facelen = len(faces)
    find = 0
    nfaces = np.empty((facelen << 1, 3), dtype=np.uint32)
    match mat:
      case 1:
        coord = uv
        mind = 0
        texmap[:] = np.asarray((0,1,2), np.uint32)
        mat = 2

    #arraylist things
    nmap = np.empty((facelen << 1, 3), dtype=np.uint32)

    #each time a new face is made, new verticies are made as well
    #only store new verticies?
    """
    vind = len(verts)
    verts = np.resize(verts, (vind << 1, 3))
    cind = len(coord)
    coord = np.resize(coord, (cind+vind, 2))
    """
    #flen = facelen
    #mlen = facelen

    empty = np.empty(3, dtype=np.uint32)
    for i in range(facelen):
      face = faces[i]
      tri = np.asarray(((verts[face[0]][0],verts[face[0]][1],verts[face[0]][2]), (verts[face[1]][0],verts[face[1]][1],verts[face[1]][2]), (verts[face[2]][0],verts[face[2]][1],verts[face[2]][2])), dtype=np.float32)
      lnpnd = lambda v, plane: (v[0] * plane[1][0] + v[1] * plane[1][1] + v[2] * plane[1][2]) - (plane[0][0] * plane[1][0] + plane[0][1] * plane[1][1] + plane[0][2] * plane[1][2])
      voutn = 0
      vinn = 0
      vout = empty.copy()
      vin = empty.copy()
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
          nfaces[find] = (face[0],face[1],face[2])
          find += 1
          match mat:
            case 2:
              nmap[mind] = (texmap[i][0],texmap[i][1],texmap[i][2])
              mind += 1
          continue
        case 1:
          nt = empty.copy()
          nt[vin[0]] = face[vin[0]]
          nt[vin[1]] = face[vin[1]]
          nv, t = intersect(tri[vout[0]], tri[vin[0]], plane[1], plane[0])
          verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
          nt[vout[0]] = len(verts) - 1
          nfaces[find] = (nt[0],nt[1],nt[2])
          find += 1
          match mat:
            case 2:
              nm = empty.copy()
              nm[vin[0]] = texmap[i, vin[0]]
              nm[vin[1]] = texmap[i, vin[1]]
              ntex = t * (coord[texmap[i, vin[0]]] - coord[texmap[i, vout[0]]]) + coord[texmap[i, vin[0]]]
              coord = np.append(coord,np.asarray([(ntex[0],ntex[1])],np.float32),axis=0)
              nm[vout[0]] = len(coord) - 1
              nmap[mind] = (nm[0],nm[1],nm[2])
              mind += 1
              nvm = empty.copy()
              nvm[vout[0]] = nm[vout[0]]
              nvm[vin[0]] = texmap[i, vin[1]]
              nvt = empty.copy()
              nvt[vout[0]] = nt[vout[0]]
              nvt[vin[0]] = face[vin[1]]
              nv, t = intersect(tri[vout[0]], tri[vin[1]], plane[1], plane[0])
              verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
              nvt[vin[1]] = nt[vout[0]] + 1
              nfaces[find] = (nvt[0],nvt[1],nvt[2])
              find += 1
              ntex = t * (coord[texmap[i, vin[1]]] - coord[texmap[i, vout[0]]]) + coord[texmap[i, vin[1]]]
              coord = np.append(coord, np.asarray([(ntex[0],ntex[1])],np.float32),axis=0)
              nvm[vin[1]] = nm[vout[0]] + 1
              nmap[mind] = (nvm[0],nvm[1],nvm[2])
              mind += 1
            case _:
              nvt = empty.copy()
              nvt[vout[0]] = nt[vout[0]]
              nvt[vin[0]] = face[vin[1]]
              nv, t = intersect(tri[vout[0]], tri[vin[1]], plane[1], plane[0])
              verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
              nvt[vin[1]] = nt[vout[0]] + 1
              nfaces[find] = (nvt[0],nvt[1],nvt[2])
              find += 1
          #print(nvt)
          #print(verts)
          #print("onedetected")

          continue
        case 2:
          nt = empty.copy()
          nt[vin[0]] = face[vin[0]]
          nv, t = intersect(tri[vout[0]], tri[vin[0]], plane[1], plane[0])
          verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
          nt[vout[0]] = len(verts)-1
          match mat:
            case 2:
              nm = empty.copy()
              nm[vin[0]] = texmap[i, vin[0]]
              ntex = t * (coord[texmap[i, vin[0]]] - coord[texmap[i, vout[0]]]) + coord[texmap[i, vin[0]]]
              coord = np.append(coord, np.asarray([(ntex[0],ntex[1])],np.float32),axis=0)
              nm[vout[0]] = len(coord) - 1
              nv, t = intersect(tri[vout[1]], tri[vin[0]], plane[1], plane[0])
              verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
              nt[vout[1]] = nt[vout[0]] + 1
              ntex = t * (coord[texmap[i, vin[0]]] - coord[texmap[i, vout[1]]]) + coord[texmap[i, vin[0]]]
              coord = np.append(coord, np.asarray([(ntex[0],ntex[1])],np.float32),axis=0)
              nm[vout[1]] = nm[vout[0]] + 1
              nmap[mind] = (nm[0],nm[1],nm[2])
              mind += 1
            case _:
              nv, t = intersect(tri[vout[1]], tri[vin[0]], plane[1], plane[0])
              verts = np.append(verts, np.asarray([(nv[0],nv[1],nv[2])]), axis=0)
              nt[vout[1]] = nt[vout[0]] + 1

          nfaces[find] = (nt[0],nt[1],nt[2])
          find += 1
          continue
        case 3:
          continue

    nfaces = nfaces[:find]
    nmap = nmap[:mind]
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
  t = (dot3d(pp - vp, pn) + 1e-32) / ((vn[0] * pn[0] + vn[1] * pn[1] + vn[2] * pn[2]) + 1e-32)
  return np.asarray((vn[0] * t + vp[0], vn[1] * t + vp[1], vn[2] * t + vp[2]),np.float32), t


#print(intersect(np.asarray([5.1,3.5,-9.2]),np.asarray([-4.9,-6.5,-5.3]),np.asarray([5,5,-4.9]),np.asarray([-1.71,20,-7.8])))
#to do clipping


@njit()
def magn(v):
  return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


@njit()
def norm(v):
  return v / math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
