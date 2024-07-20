import pygame as pg
import numpy as np
from pygame.constants import K_DOWN, KEYDOWN
from pygame.locals import QUIT
pg.init()
tex = pg.surfarray.array3d(pg.image.load("unaliased.png")).astype(np.uint8)
convtex = tex.copy().astype(np.int16)
thresthold = np.asarray((200,200,200),np.uint8)
width = len(tex)
height = len(tex[0])
screen = pg.display.set_mode((width*5, height*5))
en = True
mode = True
#edge = np.zeros((width, height), dtype="bool")
"""
for x in range(1,width-2):
  for y in range(1,height-2):
    c1 = abs(tex[x+1,y] - tex[x-1,y])
    c2 = abs(tex[x,y+1] - tex[x,y-1])
    c3 = abs(tex[x+1,y+1] - tex[x-1,y-1])
    c4 = abs(tex[x+1,y-1] - tex[x-1,y+1])
    if (c1[0] > thresthold[0] and c1[1] > thresthold[1] and c1[2] > thresthold[2]) or (c2[0] > thresthold[0] and c2[1] > thresthold[1] and c2[2] > thresthold[2]) or (c3[0] > thresthold[0] and c3[1] > thresthold[1] and c3[2] > thresthold[2]) or (c4[0] > thresthold[0] and c4[1] > thresthold[1] and c4[2] > thresthold[2]):
      edge[x,y] = True
      ntex[x,y] = (255,0,0)
"""
up = 5

while True:
  for event in pg.event.get():
    if event.type == QUIT:
      break
    if event.type == pg.KEYDOWN:
      if event.key == pg.K_ESCAPE:
        pg.quit()
        break
      elif event.key == pg.K_TAB:
        en = not en
      elif event.key == pg.K_RETURN:
        mode = not mode
      elif event.key == pg.K_EQUALS:
        up += 1
        screen = pg.display.set_mode((width*up, height*up))
      elif event.key == pg.K_MINUS:
        up -= 1
        screen = pg.display.set_mode((width*up, height*up))

  keys = pg.key.get_pressed()
  if(keys[pg.K_UP]):
    thresthold += np.asarray((1,1,1),np.uint8)
    print(thresthold)
  elif(keys[pg.K_DOWN]):
    thresthold -= np.asarray((1,1,1),np.uint8)
    print(thresthold)
    
    
  ntex = tex.copy()
  if en:
    if mode:
      for x in range(width-1):
        for y in range(height-1):
          #c1 is difference across different channels
          c1 = abs(convtex[x,y]*8-convtex[min(x+1,width-1),y]-convtex[min(x+1,width-1),min(y+1,height-1)]-convtex[min(x+1,width-1),max(y-1,0)]-convtex[x,min(y+1,height-1)]-convtex[x,max(y-1,0)]-convtex[max(x-1,0),y]-convtex[max(x-1,0),min(y+1,height-1)]-convtex[max(x-1,0),max(y-1,0)])
          #c2 = (-(-convtex[x,y]*8 + convtex[min(x+1,width-1),y] + convtex[min(x+1,width-1),min(y+1,height-1)] + convtex[min(x+1,width-1),max(y-1,0)] + convtex[x,min(y+1,height-1)] + convtex[x,max(y-1,0)] + convtex[max(x-1,0),y] + convtex[max(x-1,0),min(y+1,height-1)] + convtex[max(x-1,0),max(y-1,0)]))
          #if ((c1[0] > thresthold[0] or c1[1] > thresthold[1] or c1[2] > thresthold[2]) or (c2[0] > thresthold[0] or c2[1] > thresthold[1] or c2[2] > thresthold[2])):
          if (c1[0] > thresthold[0] or c1[1] > thresthold[1] or c1[2] > thresthold[2]):
            ntex[x,y] = (convtex[x,y]+convtex[min(x+1,width-1),y]+convtex[min(x+1,width-1),min(y+1,height-1)]+convtex[min(x+1,width-1),max(y-1,0)]+convtex[x,min(y+1,height-1)]+convtex[x,max(y-1,0)]+convtex[max(x-1,0),y]+convtex[max(x-1,0),min(y+1,height-1)]+convtex[max(x-1,0),max(y-1,0)])/9
    else:
      ntex = np.zeros((width,height,3),np.uint8)
      for x in range(width-1):
        for y in range(height-1):
          #c1 is difference across different channels
          c1 = abs(convtex[x,y]*8-convtex[min(x+1,width-1),y]-convtex[min(x+1,width-1),min(y+1,height-1)]-convtex[min(x+1,width-1),max(y-1,0)]-convtex[x,min(y+1,height-1)]-convtex[x,max(y-1,0)]-convtex[max(x-1,0),y]-convtex[max(x-1,0),min(y+1,height-1)]-convtex[max(x-1,0),max(y-1,0)])
          #c2 = (-(-convtex[x,y]*8 + convtex[min(x+1,width-1),y] + convtex[min(x+1,width-1),min(y+1,height-1)] + convtex[min(x+1,width-1),max(y-1,0)] + convtex[x,min(y+1,height-1)] + convtex[x,max(y-1,0)] + convtex[max(x-1,0),y] + convtex[max(x-1,0),min(y+1,height-1)] + convtex[max(x-1,0),max(y-1,0)]))
          #if ((c1[0] > thresthold[0] or c1[1] > thresthold[1] or c1[2] > thresthold[2]) or (c2[0] > thresthold[0] or c2[1] > thresthold[1] or c2[2] > thresthold[2])):
          if (c1[0] > thresthold[0] or c1[1] > thresthold[1] or c1[2] > thresthold[2]):
            ntex[x,y] = (255,0,0)

  #upscale
  utex = np.zeros((width*up,height*up,3),np.uint8)
  for x in range(width-1):
    for y in range(height-1):
      for i in range(up):
        for j in range(up):
          utex[x*up+i,y*up+j] = ntex[x,y]
  pg.surfarray.blit_array(screen, utex)
  pg.display.flip()