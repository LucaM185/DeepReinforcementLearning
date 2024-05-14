import numpy as np
import pygame
import math
import os
import torch
import sys
from Track import * 
from Car import Car
from Model import MLP
import multiprocessing


pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)

track_image, track_mask = get_track_mask(r"C:\Users\lucam\Desktop\VScode\Cars\monza.png")

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Car Steering on Track with Camera')
clock = pygame.time.Clock()

resolution = 18

n_cars = 18
cars = [Car(SCREEN_WIDTH // 8 + 20, SCREEN_HEIGHT // 8 + i*0 + 20, 60, 30, model=MLP(*MLP.setup), track_mask=track_mask) for i in range(n_cars)]

lastx, lasty = 0, 0

display_radar = True
display = True
running = True

# Game loop
for j in range(100):
    for i in range(600 + j*100):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()

        screen.fill(WHITE)
        screen.blit(track_image, (-lastx, -lasty))            
        
        outputs = [car.get_model_output(car.model, resolution) for car in cars]
            
        for car, output in zip(cars, outputs):
            car.accelerate()

            if output.argmax(axis=-1).item() == 0:
                car.steer_left()
            elif output.argmax(axis=-1).item() == 2:
                car.steer_right()

            car.update(track_mask)

            if not car.crashed:
                offset_x = car.x - SCREEN_WIDTH // 2
                offset_y = car.y - SCREEN_HEIGHT // 2
                output_backup = output

            if display: car.draw(screen, lastx, lasty)
            if display_radar: car.visualize(track_mask, screen, lastx, lasty, resolution=resolution)

            radar = car.get_radar_readings(track_mask, angle_range=45, resolution=resolution)
            speed = car.speed
            buttons = torch.nn.functional.one_hot(output.argmax(), 3).float()
            car.recording.append(torch.cat([torch.tensor(radar), torch.tensor([speed]), buttons]))

        if display:
            font = pygame.font.Font(None, 36)
            text = font.render(f"Output: [{output_backup[0]:.2f}, {output_backup[1]:.2f}, {output_backup[2]:.2f}]", True, (0, 0, 0))
            screen.blit(text, (10, 10))

        lastx, lasty = offset_x, offset_y
        pygame.display.flip()
        clock.tick(1000)
    
        if keys[pygame.K_q]:
            running = False
            break
        
        if [car.crashed for car in cars].count(True) == n_cars:
            break

    bestOdometer = -10
    for n, car in enumerate(cars):
        if car.odometer > bestOdometer:
            if torch.rand(1) > -1.0: 
                bestOdometer = car.odometer
                bestmodel = car.model
                bestn = n


    print(f"{cars[bestn].odometer:2f} - {speed} ")

    past = torch.randint(1, 100, (n_cars,))
    cars = [Car(SCREEN_WIDTH // 8 + 20, SCREEN_HEIGHT // 8 + 20, 60, 30, bestmodel.train(past[i].item(), 10, torch.stack(cars[i].recording, axis=0)), track_mask=track_mask)  for i in range(n_cars)]
    if running == False: break

pygame.quit()
