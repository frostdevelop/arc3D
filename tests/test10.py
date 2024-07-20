import pygame as pg
from pygame.locals import QUIT
width = 0
height = 0
pg.init()
screen = pg.display.set_mode((width, height), pg.FULLSCREEN)
info = pg.display.Info()
width = info.current_w
height = info.current_h
surface = pg.surface.Surface((width, height))

'''
pg.draw.rect(surface, (255, 0, 0), (7.74e+02, 2.82e+02, 5, 5))
pg.draw.rect(surface, (0, 255, 0), (4.65e+02, 4.76e+02, 5, 5))
pg.draw.rect(surface, (0, 0, 255), (8.29e+02, 4.76e+02, 5, 5))

#pg.draw.rect(surface, (255, 255, 0), (4.65e+02, 4.76e+02, 5, 5))

pg.draw.rect(surface, (255, 255, 0), (4.43e+02, 2.56e+02, 5, 5))

#pg.draw.rect(surface, (255, 255, 0), (7.74e+02, 2.82e+02, 5, 5))


pg.draw.rect(surface, (255, 0, 0), (3.58e+02, 5.47e+02, 5, 5))
pg.draw.rect(surface, (0, 255, 0), (9.18e+02, 3.78e+02, 5, 5))
pg.draw.rect(surface, (0, 0, 255), (9.29e+02, 5.47e+02, 5, 5))

pg.draw.rect(surface, (255, 0, 0), (3.58e+02, 5.47e+02, 5, 5))
pg.draw.rect(surface, (0, 255, 0), (5.1e+02, 3.31e+02, 5, 5))
pg.draw.rect(surface, (0, 0, 255), (9.18e+02, 3.78e+02, 5, 5))

pg.draw.rect(surface, (255, 0, 0), (3.72e+02, 5.36e+02, 5, 5))
pg.draw.rect(surface, (0, 255, 0), (9.68e+02, 3.75e+02, 5, 5))
pg.draw.rect(surface, (0, 0, 255), (1.175e+03, 5.36e+02, 5, 5))
'''

screen.blit(surface, (0, 0))
pg.display.flip()
gg = input()