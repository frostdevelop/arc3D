import numpy as np
import pygame as pg
from pygame.locals import QUIT
from arcimport import readobj
from arc import *

'''
Material Settings:
0: Color by position
1: Same texture for triangles
2: Mapped Texture
3: Fog
'''

def main():
  width = 0
  height = 0
  pg.init()
  screen = pg.display.set_mode((width, height), pg.FULLSCREEN)
  info = pg.display.Info()
  width = info.current_w
  height = info.current_h
  clock = pg.time.Clock()
  font = pg.font.Font(pg.font.get_default_font(), 30)
  pg.display.set_caption("Arc3D")
  pg.mouse.set_visible(0)

  background = pg.surface.Surface((width, height))
  background.fill((255, 255, 255))
  screen.blit(background, (0, 0))
  loading = font.render("Loading...", False, (0, 0, 0))
  screen.blit(loading, (width / 2 - loading.get_width() / 2, height / 2 - loading.get_height() / 2))
  pg.display.flip()

  scube = Object(np.asarray([[-2, -2, -2], [2, 2, -2], [-2, 2, -2],[2, -2, -2], [-2, -2, 2], [2, 2, 2], [-2, 2, 2],[2, -2, 2]], dtype=np.float32),np.array([[0, 2, 3], [2, 1, 3], [7, 5, 4], [5, 6, 4], [4, 6, 0], [6, 2, 0], [3, 1, 7], [1, 5, 7], [7, 4, 0],[0, 3, 7], [2, 6, 1], [6, 5, 1]], dtype=np.uint16),2,tex="models/sheepy.png",texcoord=[[0, 0], [0, 1], [1, 0], [1, 1]],texmap=[[0, 1, 2], [1, 3, 2], [0, 1, 2], [1, 3, 2], [0, 1, 2],[1, 3, 2], [0, 1, 2], [1, 3, 2], [2, 0, 1], [1, 3, 2],[0, 1, 2], [1, 3, 2]])

  stri = Object(np.asarray([[-2,0,-2],[0,0,-4],[2,0,-2]], dtype=np.float32),
  np.array([[0,2,1]], dtype=np.uint16),
  2,
  tex="models/sheepy.png",
  texcoord=[[0, 0], [0, 1], [1, 0], [1, 1]],
  texmap=[[0, 1, 2]])

  iverts, itris, icoord, imap = readobj("models/teapot.obj")
  teapot = Object(iverts,itris,1,tex="models/sheepy.png")
  iverts, itris, icoord, imap = readobj("models/mountains.obj")
  mountains = Object(iverts,
                     itris,
                     1,
                     tex="models/box.jpeg")
  iverts, itris, icoord, imap = readobj("models/Babycrocodile.obj")
  croc = Object(iverts, itris, 2, tex="models/BabyCrocodileGreen.png", texcoord=icoord, texmap=imap)

  iverts, itris, icoord, imap = readobj("models/pacifikytext.obj")
  pacifikytext = Object(iverts, itris, 0)

  light = Light(np.array([0, 1, 1],dtype=np.float32))

  #scene = Scene([[scube,np.asarray([10,15,10], dtype=np.float32)],[croc, np.asarray([0, 10, 0],dtype=np.float32)],[mountains, np.asarray([0, -10, 0], dtype=np.float32)],[pacifikytext,np.asarray([-10,15,-20], dtype=np.float32)]],light,[50, 127, 200])
  scene = Scene([[scube,np.asarray([10,15,10], dtype=np.float32)],[croc, np.asarray([0, 10, 0],dtype=np.float32)],[mountains, np.asarray([0, -10, 0], dtype=np.float32)]],light,[50, 127, 200])
  #scene = Scene([[scube,np.asarray([10,15,10], dtype=np.float32)],[croc, np.asarray([0, 10, 0],dtype=np.float32)]],light,[50, 127, 200])
  #scene = Scene([[stri,np.asarray([0,0,-0.5], dtype=np.float32)]],light,[50, 127, 200])
  #scene = Scene([[mountains,np.asarray([0,0,0], dtype=np.float32)]],light,[50,127,200])
  #scene = Scene([[scube,np.asarray([0,0,0], dtype=np.float32)]], light, [50,127,200])
  #camera = Camera(np.pi/8,[0,0,-10],0,0, width, height)
  camera = Camera(np.pi / 2, np.asarray([0.0, 1.5, -5.0], dtype=np.float32), 0, 0, 1000, 0.5)
  renderer = Renderer(width, height, camera, [[1, 0], [0, 1], [1, 1]], False)
  running = True

  while running:
    elapsed_time = clock.tick() * 0.001

    for event in pg.event.get():
      if event.type == pg.QUIT:
        running = False
      if event.type == pg.KEYDOWN:
        if event.key == pg.K_ESCAPE:
          running = False
        """
        if event.key == pg.K_r:
          croc.mat += 1
          if croc.mat > 3: croc.mat = 0
        """

    renderer.movement(elapsed_time)

    scene.objects[0][0].verts = renderer.rotate(scene.objects[0][0].verts, elapsed_time / 2, elapsed_time)
    scene.objects[1][0].verts = renderer.rotate(scene.objects[1][0].verts, elapsed_time/5, 0)

    light.dir = np.array([np.sin(pg.time.get_ticks() / 1000), 1, 1], dtype=np.float32)
    light.dir = light.dir / np.linalg.norm(light.dir)

    renderer.render(scene)
    surface = pg.surfarray.make_surface(renderer.surface)
    screen.blit(surface, (0, 0))
    #screen.blit(renderer.output, (0, 0))

    positiontext = font.render(
        f'XYZ: {truncate(camera.position[0])} {truncate(camera.position[1])} {truncate(camera.position[2])}',
        False, (0, 0, 0))
    angletext = font.render(
        f'Angle: {truncate(camera.yang)} {truncate(camera.xang)}', False,
        (0, 0, 0))
    fps = font.render(f'FPS: {round(1/(elapsed_time + 1e-16))}', False, (0,0,0))
    screen.blit(positiontext, (10, 10))
    screen.blit(angletext, (10, 50))
    screen.blit(fps, (10, 90))
    pg.display.flip()

if __name__ == "__main__":
  main()
  pg.quit()