import pygame as pg
import numpy as np
from numba import njit, jit

width = height = 0
pg.init()
screen = pg.display.set_mode((width, height), pg.FULLSCREEN)
pg.display.set_caption("Triangles")
info = pg.display.Info()
width = info.current_w
height = info.current_h
print(f"{width} {height}")
running = True
tri = np.asarray([[100,300],[400,400],[200,50]],np.int32)
RED, GREEN, BLUE = [255,0,0], [0,255,0], [0,0,255]

tex = pg.surfarray.array3d(pg.image.load("models/pfpatlas.png")).astype(np.uint8)
uv = np.asarray([[0.5,0], [0,1], [1,1]]).astype(np.float32)
texsize = [len(tex)-1, len(tex[0])-1]

def ren(frame, tri, tex, uv, texsize):
  ysort = np.argsort(tri[:, 1])
  xstart, ystart = tri[ysort[0]]
  xmiddle, ymiddle = tri[ysort[1]]
  xend, yend = tri[ysort[2]]

  xslope1 = (xend-xstart)/(yend-ystart + 1e-16)
  xslope2 = (xmiddle-xstart)/(ymiddle-ystart + 1e-16)
  xslope3 = (xend-xmiddle)/(yend-ymiddle + 1e-16)

  uvstart = uv[ysort[0]]
  uvmiddle = uv[ysort[1]]
  uvend = uv[ysort[2]]

  uvslope1 = (uvend-uvstart)/(yend-ystart + 1e-16)
  uvslope2 = (uvmiddle-uvstart)/(ymiddle-ystart + 1e-16)
  uvslope3 = (uvend-uvmiddle)/(yend-ymiddle + 1e-16)

  for y in range(ystart, yend):
    x1 = xstart + int(xslope1*(y-ystart))
    uv1 = uvstart + (y-ystart)*uvslope1

    if y < ymiddle:
      x2 = xstart + int(xslope2*(y-ystart))
      uv2 = uvstart + (y-ystart)*uvslope2
    else:
      x2 = xmiddle + int(xslope3*(y-ymiddle))
      uv2 = uvmiddle + (y-ymiddle)*uvslope3

    if x1 > x2:
      x1, x2 = x2, x1
      uv1, uv2 = uv2, uv1

    uvslope = (uv2-uv1)/(x2-x1 + 1e-16)

    for x in range(x1, x2):
      uvo = uv1 + (x-x1)*uvslope
      frame[x, y] = tex[int(uvo[0]*texsize[0]), int(uvo[1]*texsize[1])]

def ren(frame, tri, tex, uv, texsize):
  ysort = np.argsort(tri[:, 1])
  xstart, ystart = tri[ysort[0]]
  xmiddle, ymiddle = tri[ysort[1]]
  xend, yend = tri[ysort[2]]

  xslope1 = (xend-xstart)/(yend-ystart + 1e-16)
  xslope2 = (xmiddle-xstart)/(ymiddle-ystart + 1e-16)
  xslope3 = (xend-xmiddle)/(yend-ymiddle + 1e-16)

  uvstart = uv[ysort[0]]
  uvmiddle = uv[ysort[1]]
  uvend = uv[ysort[2]]

  uvslope1 = (uvend-uvstart)/(yend-ystart + 1e-16)
  uvslope2 = (uvmiddle-uvstart)/(ymiddle-ystart + 1e-16)
  uvslope3 = (uvend-uvmiddle)/(yend-ymiddle + 1e-16)

  for y in range(min(ystart,height), min(yend,height)):
    x1 = xstart + xslope1*(y-ystart)
    uv1 = uvstart + (y-ystart)*uvslope1

    if y < ymiddle:
      x2 = xstart + xslope2*(y-ystart)
      uv2 = uvstart + (y-ystart)*uvslope2
    else:
      x2 = xmiddle + xslope3*(y-ymiddle)
      uv2 = uvmiddle + (y-ymiddle)*uvslope3

    if x1 > x2:
      x1, x2 = x2, x1
      uv1, uv2 = uv2, uv1

    uvslope = (uv2-uv1)/(x2-x1 + 1e-16)

    for x in range(min(int(x1),width), min(int(x2),width)):
      uvo = uv1 + (x-min(int(x1),width))*uvslope
      if x == int(x1):
        p = x1 - int(x1)
        print(p)
        frame[x, y] = p*frame[x, y] + (1-p)*tex[int(uvo[0]*texsize[0]), int(uvo[1]*texsize[1])]
      if x == int(x2):
        p = x2 - int(x2)
        print(p)
        frame[x, y] = p*frame[x, y] + (1-p)*tex[int(uvo[0]*texsize[0]), int(uvo[1]*texsize[1])]
      frame[x, y] = tex[int(uvo[0]*texsize[0]), int(uvo[1]*texsize[1])]
      

while running:
  for event in pg.event.get():
    if event.type == pg.QUIT:
      running = False
    if event.type == pg.KEYDOWN:
      if event.key == pg.K_ESCAPE: running = False
      if event.key == pg.K_SPACE: tri = np.random.randint(0, height/2, (3, 2))

  frame = np.zeros((width, height, 3), dtype=np.uint8)
  frame[:,:] = (40,40,40)
  ren(frame, tri, tex, uv, texsize)
  #fxaa
  for x in range(width):
    for y in range(height):
      
    
  #frame[x1:x2,y] = color

  surf = pg.surfarray.make_surface(frame)
  screen.blit(surf, (0,0))
  pg.display.update()

pg.quit()
