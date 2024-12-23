import numpy as np
import math
import pygame as pg
from pygame.locals import QUIT
from arcimport import readobj
from arcdev import *

def main():
  pg.init()
  screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
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
  #try:
  iverts, itris, icoord, imap = readobj("models/sponzaoneImproved.obj")
  sponza = Object(iverts,itris,1,[0, -10, -5],[0,0],tex="models/sponza_diff-0.25.png")
  #except e:
      #print(e)
  
  light = Light(np.array([0, 1, 1],dtype=np.float32))

  scene = Scene([sponza],light,(50, 127, 200))
  camera = Camera(np.pi / 2, (0.0, 1.5, -5.0), (0,0), 1000, 0.5)
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

    renderer.move(elapsed_time)
    #print(camera.velocity)
    
    #"""

    light.upd((math.sin(pg.time.get_ticks() / 1000), 1, 1))
    #"""

    renderer.render(scene)
    screen.blit(pg.surfarray.make_surface(renderer.surface), (0, 0))

    #"""
    positiontext = font.render(
        f'XYZ: {truncate(camera.position[0])} {truncate(camera.position[1])} {truncate(camera.position[2])}',
        False, (0, 0, 0))
    angletext = font.render(
        f'Angle: {truncate(camera.angle[1])} {truncate(camera.angle[0])}', False,
        (0, 0, 0))
    fps = font.render(f'FPS: {round(1/(elapsed_time + 1e-16))}', False, (0,0,0))
    screen.blit(positiontext, (10, 10))
    screen.blit(angletext, (10, 50))
    screen.blit(fps, (10, 90))
    #"""
    pg.display.flip()

if __name__ == "__main__":
  main()
  print("Quit...")
  pg.quit()
