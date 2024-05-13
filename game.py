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


setup = [13, 3, 16, 1]

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)

track_image, track_mask = get_track_mask(r"C:\Users\lucam\Desktop\VScode\Cars\monza.png")

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Car Steering on Track with Camera')
clock = pygame.time.Clock()

resolution = 12

n_cars = 18
cars = [Car(SCREEN_WIDTH // 8 + 20, SCREEN_HEIGHT // 8 + i*0 + 20, 60, 30, model=MLP(*setup), track_mask=track_mask) for i in range(n_cars)]

lastx, lasty = 0, 0

display_radar = False
display = True
running = True

def process_item(car: Car, resolution: int):
    # Run the slow line for each car
    output = car.get_model_output(car.model, resolution)
    return output


# Game loop
for j in range(100):
    for i in range(600 + j*100):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if display:
            screen.fill(WHITE)
            screen.blit(track_image, (-lastx, -lasty))
            
        keys = pygame.key.get_pressed()
        
        
        # num_processes = multiprocessing.cpu_count()

        # # Create a pool of processes
        # pool = multiprocessing.Pool(processes=num_processes)

        # Map the function to process each car to the pool of processes

        outputs = [process_item(car, resolution) for car in cars]
            
        for car, output in zip(cars, outputs):
            

            car.accelerate()
            #output += torch.rand(3) * 0.1
            # print(output.argmax(axis=-1))
            if output.argmax(axis=-1).item() == 0:
                car.steer_left()
            elif output.argmax(axis=-1).item() == 2:
                car.steer_right()


            # Update the car
            car.update(track_mask)

            # Camera offset
            if not car.crashed:
                offset_x = car.x - SCREEN_WIDTH // 2
                offset_y = car.y - SCREEN_HEIGHT // 2
                output_backup = output

            if display:
                car.draw(screen, lastx, lasty)
                if display_radar:
                    car.visualize(track_mask, screen, lastx, lasty, resolution=resolution)

            radar = car.get_radar_readings(track_mask, angle_range=45, resolution=resolution)
            speed = car.speed
            buttons = torch.nn.functional.one_hot(output.argmax(), 3).float()
            car.recording.append(torch.cat([torch.tensor(radar), torch.tensor([speed]), buttons]))

        # draw output round2 on top left of the screen 
        # monospace font
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

    past = []
    for past_inter in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 90, 120, 150]:
        for epochs in [10]:
            past.append((past_inter, epochs))
    past = past[::-1]

    # for k in range(1):
    #     for i in range(n_cars):
    #         recording = torch.stack(cars[i].recording, axis=0)
    #         bestmodel = bestmodel.train(*[10, 3], recording)

    cars = [Car(SCREEN_WIDTH // 8 + 20, SCREEN_HEIGHT // 8 + 20 + torch.randint(0, 15, (1,)).item()*0, 60, 30, bestmodel.train(*past[i], torch.stack(cars[bestn].recording, axis=0)) if torch.rand(1).item() > -0.1 else bestmodel.train(*past[i], torch.stack(cars[i].recording, axis=0)), track_mask=track_mask)  for i in range(n_cars)]

    if running == False: break

pygame.quit()
