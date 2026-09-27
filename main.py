import numpy as np
import math
import pygame as pg
from pygame.locals import QUIT
from arcimport import readobj, read_obj
from arcdev import *

'''
Material Settings:
0: Color by position
1: Same texture for triangles
2: Texture
3: Fog
4: Solid Color
5: Glowing Color
6: Glowing Texture
'''

def main():
  print("Initializing...")
  
  pg.init()
  screen = pg.display.set_mode((1280, 720), pg.RESIZABLE)
  info = pg.display.Info()
  dimensions = [info.current_w,info.current_h]
  clock = pg.time.Clock()
  font = pg.font.Font(pg.font.get_default_font(), 25)
  font_lg = pg.font.Font(pg.font.get_default_font(), 35)
  pg.display.set_caption("Arc3D Demo")
  pg.mouse.set_visible(0)
  pg.display.set_icon(pg.image.load('brand/arc3d.png'))
  
  selections = ["Testing", "Torus", "Sponza", "Toad", "Hall", "Leave"]
  padding = 20
  running = True
  is_fullscreen = False
  selection = 0
  while running:
      selecting = True
      while selecting:
        elapsed_time = clock.tick() * 0.001

        for event in pg.event.get():
          if event.type == pg.QUIT:
            return
          elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
              selecting = False
            elif event.key == pg.K_ESCAPE:
              return;
            elif event.key == pg.K_F11:
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    screen = pg.display.set_mode((1280, 720), pg.FULLSCREEN)
                    dimensions[0] = info.current_w
                    dimensions[1] = info.current_h
                else:
                    screen = pg.display.set_mode((info.current_w, info.current_h), pg.RESIZABLE)
                    dimensions[0] = info.current_w
                    dimensions[1] = info.current_h
            elif event.key == pg.K_UP:
                selection = (selection-1) % len(selections)
            elif event.key == pg.K_DOWN:
                selection = (selection+1) % len(selections)
          elif event.type == pg.VIDEORESIZE and not is_fullscreen:
            screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
            dimensions[0] = event.w
            dimensions[1] = event.h

        screen.fill((0, 0, 0))
        selection_texts = []
        selection_width = 400
        selection_height = padding
        title_text = font_lg.render("Arc3D Demos", True, (255,255,0))
        title_text_height = title_text.get_height()
        selection_height += title_text_height+int(padding*2.5)
        title_text_width = title_text.get_width()
        selection_width = max(selection_width, title_text_width+padding*2)
         
        for i in range(len(selections)):
            text = font.render((">" if i == selection else "")+selections[i], True, ((255,255,0) if i == selection else (200,200,200)))
            selection_texts.append((text,selection_height))
            selection_height += text.get_height()+padding
            selection_width = max(selection_width, text.get_width()+padding*2)
        
        selection_surface = pg.Surface((selection_width,selection_height))
        pg.draw.rect(selection_surface, (255,255,0), pg.Rect(0,0,selection_width,selection_height), border_radius=10, width=2)
        selection_surface.blit(title_text, (selection_width/2 - title_text_width/2,padding))
        pg.draw.rect(selection_surface, (255,255,0), pg.Rect(0,padding*2+title_text_height-2,selection_width,2))
        for text in selection_texts:
            selection_surface.blit(text[0], (padding,text[1]))
        
        screen.blit(selection_surface, (int(dimensions[0]/2 - selection_width/2),int(dimensions[1]/2 - selection_height/2)))
        
        pg.display.flip()
        
      print(f"Selected {selections[selection]}")
      match(selection):
        case 0:
            testing(screen,info,dimensions,clock,font,font_lg,is_fullscreen)
        case 1:
            torus(screen,info,dimensions,clock,font,font_lg,is_fullscreen)
        case 2:
            sponza(screen,info,dimensions,clock,font,font_lg,is_fullscreen)
        case 3:
            toad(screen,info,dimensions,clock,font,font_lg,is_fullscreen)
        case 4:
            hall(screen,info,dimensions,clock,font,font_lg,is_fullscreen)
        case _:
            return
        
      is_fullscreen = bool(screen.get_flags() & pg.FULLSCREEN)

def testing(screen,info,dimensions,clock,font,font_lg,is_fullscreen):
  compiled = False
  screen.fill((0, 0, 0))
  loading = font_lg.render("Parsing Model...", True, (255, 255, 0))
  screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
  pg.display.flip()

  scube = Object(np.asarray([[-2, -2, -2], [2, 2, -2], [-2, 2, -2],[2, -2, -2], [-2, -2, 2], [2, 2, 2], [-2, 2, 2],[2, -2, 2]], dtype=np.float32),np.array([[0, 2, 3], [2, 1, 3], [7, 5, 4], [5, 6, 4], [4, 6, 0], [6, 2, 0], [3, 1, 7], [1, 5, 7], [7, 4, 0],[0, 3, 7], [2, 6, 1], [6, 5, 1]], dtype=np.uint16),2,[10,15,10],[0,0],tex="models/sheepy.png",texcoord=[[0, 0], [0, 1], [1, 0], [1, 1]],texmap=[[0, 1, 2], [1, 3, 2], [0, 1, 2], [1, 3, 2], [0, 1, 2],[1, 3, 2], [0, 1, 2], [1, 3, 2], [2, 0, 1], [1, 3, 2],[0, 1, 2], [1, 3, 2]])
  
  #stri = Object(np.asarray([[-2,0,-2],[0,0,-4],[2,0,-2]], dtype=np.float32),np.array([[0,2,1]], dtype=np.uint16),2,tex="models/sheepy.png",texcoord=[[0, 0], [0, 1], [1, 0], [1, 1]],texmap=[[0, 1, 2]])
  
  #iverts, itris, icoord, imap = readobj("models/teapot.obj")
  #teapot = Object(iverts,itris,1,tex="models/sheepy.png")
  iverts, itris, icoord, imap = readobj("models/mountains.obj")
  mountains = Object(iverts,itris,1,[0, -10, -5],[0,0],tex="models/box.jpeg")
  iverts, itris, icoord, imap = readobj("models/Babycrocodile.obj")
  croc = Object(iverts, itris, 2, [0, 10, 0], [0,0], tex="models/BabyCrocodileGreen.png", texcoord=icoord, texmap=imap)

  #iverts, itris, icoord, imap = readobj("models/pacifikytext.obj")
  #pacifikytext = Object(iverts, itris, 0)
  
  light = Light(np.array([0, 1, 1],dtype=np.float32))

  scene = Scene([scube,croc,mountains],light,(50, 127, 200),(50,200))
  #camera = Camera(np.pi/8,[0,0,-10],0,0, width, height)
  camera = Camera(np.pi / 2, (0.0, 1.5, -5.0), (0,0), 1000, 0.5)
  #camera = Camera(np.pi / 1.1, (0.0, 1.5, -5.0), (0,0), 1000, 0.5)
  renderer = Renderer(dimensions[0], dimensions[1], camera, [[1, 0], [0, 1], [1, 1]], False)
  running = True

  print("Models loaded.")
  
  screen.fill((0, 0, 0))
  loading = font_lg.render("Compiling...", True, (255, 255, 0))
  screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
  pg.display.flip()
  
  while running:
    elapsed_time = clock.tick() * 0.001

    for event in pg.event.get():
      if event.type == pg.QUIT:
        running = False
      elif event.type == pg.KEYDOWN:
        if event.key == pg.K_ESCAPE:
          running = False
        elif event.key == pg.K_F11:
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                screen = pg.display.set_mode((1280, 720), pg.FULLSCREEN)
            else:
                screen = pg.display.set_mode((info.current_w, info.current_h), pg.RESIZABLE)
            dimensions[0] = info.current_w
            dimensions[1] = info.current_h
            
            renderer.set_dimensions(dimensions)
      elif event.type == pg.VIDEORESIZE and not is_fullscreen:
        screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
        dimensions[0] = event.w
        dimensions[1] = event.h
        
        renderer.set_dimensions(dimensions)

    renderer.move(elapsed_time)
    
    scene.objects[0].rotation += elapsed_time / 2, elapsed_time
    scene.objects[0].upd()
    scene.objects[1].rotation += elapsed_time/5, 0
    scene.objects[1].upd()

    light.upd((math.sin(pg.time.get_ticks() / 1000), 1, 1))

    renderer.render(scene)
    pg.surfarray.blit_array(screen, renderer.surface)

    positiontext = font.render(
        f'XYZ: {truncate(camera.position[0])} {truncate(camera.position[1])} {truncate(camera.position[2])}',
        False, (255,255,255))
    angletext = font.render(
        f'Angle: {truncate(camera.angle[0])} {truncate(camera.angle[1])}', False, (255,255,255))
    fps = font.render(f'FPS: {round(1/(elapsed_time + 1e-16))}', False, (255,255,255))
    text_surface = pg.Surface((max(positiontext.get_width(),angletext.get_width(),fps.get_width())+20, 130))
    text_surface.set_alpha(128)
    text_surface.blit(positiontext, (10, 10))
    text_surface.blit(angletext, (10, 50))
    text_surface.blit(fps, (10, 90))    
    screen.blit(text_surface, (0,0))
    pg.display.flip()
    
    if not compiled:
        print("Render functions compiled.")
        compiled = True

def torus(screen,info,dimensions,clock,font,font_lg,is_fullscreen):
    objects = []

    screen.fill((0, 0, 0))
    loading = font_lg.render("Parsing Model...", True, (255, 255, 0))
    screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
    obj_objects, obj_materials = read_obj("models/torus_map/model.obj")
    
    screen.fill((0, 0, 0))
    loading = font_lg.render("Loading Map...", True, (255, 255, 0))
    screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
    objects.append(Object(
        obj_objects["Columns"]["verts"],
        obj_objects["Columns"]["tris"],
        2,
        [0, 0, 0],
        [0,0],
        tex=obj_materials[obj_objects["Columns"]["mat"]]["tex_diff"],
        texcoord=obj_objects["Columns"]["tex_coord"],
        texmap=obj_objects["Columns"]["tex_map"]
    ))
    
    screen.fill((0, 0, 0))
    loading = font_lg.render("Loading Torus...", True, (255, 255, 0))
    screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
    pg.display.flip()
    objects.append(Object(
        obj_objects["Torus"]["verts"],
        obj_objects["Torus"]["tris"],
        6,
        [-28.0805, 64, -190.792],
        [0,0],
        tex=obj_materials[obj_objects["Torus"]["mat"]]["tex_diff"],
        texcoord=obj_objects["Torus"]["tex_coord"],
        texmap=obj_objects["Torus"]["tex_map"]
    ))

    light = Light(np.array([0, 1, 1],dtype=np.float32))

    scene = Scene(objects,light,(0, 0, 0),(20,100))
    camera = Camera(np.pi / 2, (-28.0805, 60.5462, -170.792), (0,np.pi), 1000, 0.5)
    renderer = Renderer(dimensions[0], dimensions[1], camera, [[1, 0], [0, 1], [1, 1]], True)
    running = True

    print("Models loaded.")

    screen.fill((0, 0, 0))
    loading = font_lg.render("Compiling...", True, (255, 255, 0))
    screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
    pg.display.flip()

    collected = False
    scene.objects[1].opacity = 0.999
    
    while running:
        elapsed_time = clock.tick() * 0.001

        for event in pg.event.get():
          if event.type == pg.QUIT:
            running = False
          elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
              running = False
            elif event.key == pg.K_F11:
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    screen = pg.display.set_mode((1280, 720), pg.FULLSCREEN)
                else:
                    screen = pg.display.set_mode((info.current_w, info.current_h), pg.RESIZABLE)
                dimensions[0] = info.current_w
                dimensions[1] = info.current_h
                
                renderer.set_dimensions(dimensions)
          elif event.type == pg.VIDEORESIZE and not is_fullscreen:
            screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
            dimensions[0] = event.w
            dimensions[1] = event.h
            
            renderer.set_dimensions(dimensions)

        renderer.move(elapsed_time)
        light.upd((math.sin(pg.time.get_ticks() / 1000), 1, 1))

        renderer.render(scene)
        screen.blit(pg.surfarray.make_surface(renderer.surface), (0, 0))

        billboard_verts = renderer.projectf(np.asarray([[-28.0805, 60.5462, -190.792]], dtype=np.float32),renderer.projection, renderer.centerx,renderer.centery,renderer.camera.position,renderer.camera.angle[1],renderer.camera.angle[0])
        if billboard_verts[0][2] < 10 and not collected and (billboard_verts[0][0] < dimensions[0] and billboard_verts[0][0] > 0 and billboard_verts[0][1] < dimensions[1] and billboard_verts[0][1] > 0):
            keys = pg.key.get_pressed()
            
            text = font.render("Hold (T) to collect", False, (255,255,255))
            text_height = text.get_height()
            text_width = text.get_width()
            billboard_width = text_width+20
            billboard_height = text_height+20
            
            if keys[pg.K_t]:
                billboard_width += billboard_height
                holding_time += elapsed_time
                if(holding_time > 1):
                    collected = True
            else:
                holding_time = 0
            
            billboard_surface = pg.Surface((billboard_width,billboard_height), pg.SRCALPHA)
            pg.draw.rect(billboard_surface, (0,0,0,200), pg.Rect(0,0,billboard_width,billboard_height), border_radius=20)
            billboard_surface.blit(text,(10,10))
            pg.draw.arc(billboard_surface, (255,255,255), pg.Rect(text_width+20, 10, text_height, text_height), np.pi*0.5 - holding_time*np.pi*2, np.pi*0.5, 3)
            billboard_verts[0][0] -= billboard_width/2
            billboard_verts[0][1] -= billboard_height/2
            screen.blit(billboard_surface, (int(billboard_verts[0][0]),int(billboard_verts[0][1])))
            

        scene.objects[1].rotation += elapsed_time / 2, elapsed_time
        scene.objects[1].upd()
        for i in range(len(scene.objects[1].texcoord)):
            scene.objects[1].texcoord[i] += elapsed_time / 2
        
        positiontext = font.render(
            f'XYZ: {truncate(camera.position[0])} {truncate(camera.position[1])} {truncate(camera.position[2])}',
            False, (255,255,255))
        angletext = font.render(
            f'Angle: {truncate(camera.angle[0])} {truncate(camera.angle[1])}', False, (255,255,255))
        fps = font.render(f'FPS: {round(1/(elapsed_time + 1e-16))}', False, (255,255,255))
        text_surface = pg.Surface((max(positiontext.get_width(),angletext.get_width(),fps.get_width())+20, 130))
        text_surface.set_alpha(128)
        text_surface.blit(positiontext, (10, 10))
        text_surface.blit(angletext, (10, 50))
        text_surface.blit(fps, (10, 90))
        screen.blit(text_surface, (0,0))
        collected_display = "Collect the torus"
        if collected:
            collected_display = "Collected"
            if scene.objects[1].opacity > 0:
                collected_display = "Collected"
                scene.objects[1].opacity = np.maximum(scene.objects[1].opacity - elapsed_time,0)
        
        collected_text = font.render(
            collected_display, False,
            (255, 255, 255))
        collected_text.set_alpha(128)
        screen.blit(collected_text, (dimensions[0]-10-collected_text.get_width(), 10))
        pg.display.flip()


def hall(screen,info,dimensions,clock,font,font_lg,is_fullscreen):
  screen.fill((0, 0, 0))
  loading = font_lg.render("Parsing Model...", True, (255, 255, 0))
  screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
  pg.display.flip()
  objects = []
  
  obj_objects, obj_materials = read_obj("models/hall/LargeHall.obj")
  for object in obj_objects:
    screen.fill((0, 0, 0))
    loading = font_lg.render(f"Loading {object}...", True, (255, 255, 0))
    screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
    pg.display.flip()
    if "tex_diff" in obj_materials[obj_objects[object]["mat"]]:
        objects.append(Object(
            obj_objects[object]["verts"],
            obj_objects[object]["tris"],
            2,
            [0, 0, 0],
            [0,0],
            tex=obj_materials[obj_objects[object]["mat"]]["tex_diff"],
            texcoord=obj_objects[object]["tex_coord"],
            texmap=obj_objects[object]["tex_map"]
        ))
    else:
        if obj_objects[object]["mat"] == "Part1Mtl":
            objects.append(Object(
                obj_objects[object]["verts"],
                obj_objects[object]["tris"],
                5,
                [0, 0, 0],
                [0,0],
                color_hdr=obj_materials[obj_objects[object]["mat"]]["diff"]
            ))
        else:
            objects.append(Object(
                obj_objects[object]["verts"],
                obj_objects[object]["tris"],
                4,
                [0, 0, 0],
                [0,0],
                color_hdr=obj_materials[obj_objects[object]["mat"]]["diff"]
            ))
  
  light = Light(np.array([0, 1, 0],dtype=np.float32))

  scene = Scene(objects,light,(0,0,0),(30,100))
  camera = Camera(np.pi / 2, (-60, 8, 5), (0,-np.pi/2), 1000, 0.5)
  renderer = Renderer(dimensions[0], dimensions[1], camera, [[1, 0], [0, 1], [1, 1]], True)
  running = True
  
  print("Models loaded.")
  
  screen.fill((0, 0, 0))
  loading = font_lg.render("Compiling...", True, (255, 255, 0))
  screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
  pg.display.flip()
  
  while running:
    elapsed_time = clock.tick() * 0.001

    events = pg.event.get()
    
    for event in events:
      if event.type == pg.QUIT:
        running = False
      elif event.type == pg.KEYDOWN:
        if event.key == pg.K_ESCAPE:
          running = False
        elif event.key == pg.K_F11:
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                screen = pg.display.set_mode((1280, 720), pg.FULLSCREEN)
            else:
                screen = pg.display.set_mode((info.current_w, info.current_h), pg.RESIZABLE)
            dimensions[0] = info.current_w
            dimensions[1] = info.current_h
            
            renderer.set_dimensions(dimensions)
      elif event.type == pg.VIDEORESIZE and not is_fullscreen:
        screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
        dimensions[0] = event.w
        dimensions[1] = event.h
        
        renderer.set_dimensions(dimensions)

    renderer.move(elapsed_time)

    renderer.render(scene)
    screen.blit(pg.surfarray.make_surface(renderer.surface), (0, 0))
    
    positiontext = font.render(
        f'XYZ: {truncate(camera.position[0])} {truncate(camera.position[1])} {truncate(camera.position[2])}',
        False, (255,255,255))
    angletext = font.render(
        f'Angle: {truncate(camera.angle[0])} {truncate(camera.angle[1])}', False, (255,255,255))
    fps = font.render(f'FPS: {round(1/(elapsed_time + 1e-16))}', False, (255,255,255))
    text_surface = pg.Surface((max(positiontext.get_width(),angletext.get_width(),fps.get_width())+20, 130))
    text_surface.set_alpha(128)
    text_surface.blit(positiontext, (10, 10))
    text_surface.blit(angletext, (10, 50))
    text_surface.blit(fps, (10, 90))
    screen.blit(text_surface, (0,0))
    pg.display.flip()
    
def sponza(screen,info,dimensions,clock,font,font_lg,is_fullscreen):
  screen.fill((0, 0, 0))
  loading = font_lg.render("Parsing Model...", True, (255, 255, 0))
  screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
  pg.display.flip()
  
  iverts, itris, icoord, imap = readobj("models/sponzaoneDecimatedScaled.obj")
  sponza = Object(iverts,itris,2,[0, -10, -5],[0,0],tex="models/sponza_diff-0.5.png",texcoord=icoord, texmap=imap)

  light = Light(np.array([0, 1, 1],dtype=np.float32))

  scene = Scene([sponza],light,(50, 127, 200),(50,200))
  camera = Camera(np.pi / 2, (0.0, 1.5, -5.0), (0,0), 1000, 0.5)
  renderer = Renderer(dimensions[0], dimensions[1], camera, [[1, 0], [0, 1], [1, 1]], True)
  running = True
  
  print("Models loaded.")
  
  screen.fill((0, 0, 0))
  loading = font_lg.render("Compiling...", True, (255, 255, 0))
  screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
  pg.display.flip()
  
  while running:
    elapsed_time = clock.tick() * 0.001

    for event in pg.event.get():
      if event.type == pg.QUIT:
        running = False
      if event.type == pg.KEYDOWN:
        if event.key == pg.K_ESCAPE:
          running = False
        elif event.key == pg.K_F11:
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                screen = pg.display.set_mode((1280, 720), pg.FULLSCREEN)
            else:
                screen = pg.display.set_mode((info.current_w, info.current_h), pg.RESIZABLE)
            dimensions[0] = info.current_w
            dimensions[1] = info.current_h
            
            renderer.set_dimensions(dimensions)
      elif event.type == pg.VIDEORESIZE and not is_fullscreen:
        screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
        dimensions[0] = event.w
        dimensions[1] = event.h
        
        renderer.set_dimensions(dimensions)

    renderer.move(elapsed_time)
    light.upd((math.sin(pg.time.get_ticks() / 1000), 1, 1))

    renderer.render(scene)
    screen.blit(pg.surfarray.make_surface(renderer.surface), (0, 0))
    
    positiontext = font.render(
        f'XYZ: {truncate(camera.position[0])} {truncate(camera.position[1])} {truncate(camera.position[2])}',
        False, (255,255,255))
    angletext = font.render(
        f'Angle: {truncate(camera.angle[0])} {truncate(camera.angle[1])}', False, (255,255,255))
    fps = font.render(f'FPS: {round(1/(elapsed_time + 1e-16))}', False, (255,255,255))
    text_surface = pg.Surface((max(positiontext.get_width(),angletext.get_width(),fps.get_width())+20, 130))
    text_surface.set_alpha(128)
    text_surface.blit(positiontext, (10, 10))
    text_surface.blit(angletext, (10, 50))
    text_surface.blit(fps, (10, 90))
    screen.blit(text_surface, (0,0))
    pg.display.flip()


def toad(screen,info,dimensions,clock,font,font_lg,is_fullscreen):
  screen.fill((0, 0, 0))
  loading = font_lg.render("Parsing Model...", True, (255, 255, 0))
  screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
  pg.display.flip()
  objects = []
  
  obj_objects, obj_materials = read_obj("models/toad/model.obj")
  for object in obj_objects:
    screen.fill((0, 0, 0))
    loading = font_lg.render(f"Loading {object}...", True, (255, 255, 0))
    screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
    pg.display.flip()
    if "tex_diff" in obj_materials[obj_objects[object]["mat"]]:
        objects.append(Object(
            obj_objects[object]["verts"],
            obj_objects[object]["tris"],
            2,
            [0, 0, 0],
            [0,0],
            tex=obj_materials[obj_objects[object]["mat"]]["tex_diff"],
            texcoord=obj_objects[object]["tex_coord"],
            texmap=obj_objects[object]["tex_map"]
        ))
    else:
        objects.append(Object(
            obj_objects[object]["verts"],
            obj_objects[object]["tris"],
            4,
            [0, 0, 0],
            [0,0],
            color_hdr=obj_materials[obj_objects[object]["mat"]]["diff"]
        ))
  
  light = Light(np.array([0, 1, 1],dtype=np.float32))

  scene = Scene(objects,light,(50, 127, 200),(100,200))
  camera = Camera(np.pi / 2, (0.0, 50, 50), (0,np.pi), 1000, 0.5)
  renderer = Renderer(dimensions[0], dimensions[1], camera, [[1, 0], [0, 1], [1, 1]], True)
  running = True
  
  print("Models loaded.")
  
  screen.fill((0, 0, 0))
  loading = font_lg.render("Compiling...", True, (255, 255, 0))
  screen.blit(loading, (dimensions[0] / 2 - loading.get_width() / 2, dimensions[1] / 2 - loading.get_height() / 2))
  pg.display.flip()
  
  while running:
    elapsed_time = clock.tick() * 0.001

    events = pg.event.get()
    
    for event in events:
      if event.type == pg.QUIT:
        running = False
      elif event.type == pg.KEYDOWN:
        if event.key == pg.K_ESCAPE:
          running = False
        elif event.key == pg.K_F11:
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                screen = pg.display.set_mode((1280, 720), pg.FULLSCREEN)
            else:
                screen = pg.display.set_mode((info.current_w, info.current_h), pg.RESIZABLE)
            dimensions[0] = info.current_w
            dimensions[1] = info.current_h
            
            renderer.set_dimensions(dimensions)
      elif event.type == pg.VIDEORESIZE and not is_fullscreen:
        screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
        dimensions[0] = event.w
        dimensions[1] = event.h
        
        renderer.set_dimensions(dimensions)

    renderer.move(elapsed_time)
    light.upd((math.sin(pg.time.get_ticks() / 1000), 1, 1))

    renderer.render(scene)
    screen.blit(pg.surfarray.make_surface(renderer.surface), (0, 0))
    
    positiontext = font.render(
        f'XYZ: {truncate(camera.position[0])} {truncate(camera.position[1])} {truncate(camera.position[2])}',
        False, (255,255,255))
    angletext = font.render(
        f'Angle: {truncate(camera.angle[0])} {truncate(camera.angle[1])}', False, (255,255,255))
    fps = font.render(f'FPS: {round(1/(elapsed_time + 1e-16))}', False, (255,255,255))
    text_surface = pg.Surface((max(positiontext.get_width(),angletext.get_width(),fps.get_width())+20, 130))
    text_surface.set_alpha(128)
    text_surface.blit(positiontext, (10, 10))
    text_surface.blit(angletext, (10, 50))
    text_surface.blit(fps, (10, 90))
    screen.blit(text_surface, (0,0))
    pg.display.flip()

if __name__ == "__main__":
  main()
  print("Quit...")
  pg.quit()