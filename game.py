import numpy as np
import pygame
import math
from Car import *

setup = [9, 3, 16, 1]

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Car Steering on Track with Camera')


# Clock to control the frame rate
clock = pygame.time.Clock()


import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, n_layers):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fcx = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]) # this is a list of linear layers
        self.fc2 = nn.Linear(hidden_size, out_size)
    
    def forward(self, inputs):
        x = F.gelu(self.fc1(inputs))
        for hidden in self.fcx:    # iterating over hidden layers
            x = F.gelu(hidden(x))  # applying each hidden layer
        return torch.softmax(self.fc2(x), axis=-1)

    def train(self, past_interaction = 10, epochs = 50, recording=None): 
        past_interaction *= 1
        bestmodel = self

        # make labels so that buttons that lead to high speed 20 steps after are rewarded
        buttons = recording[:, -3:]
        radar = recording[:, :9]
        #speedthresh = 1.5
        #buttons[speed < speedthresh] = (buttons[speed < speedthresh] < 0.5) + 0.0

        crashed_timestamp = radar.argmin(-1) if (radar[radar.argmin(-1)].sum() == 0) else -1 
        buttons = buttons[:crashed_timestamp]
        radar = radar[:crashed_timestamp]

        buttons[:-past_interaction] = 1-buttons[:-past_interaction]

        # train model   
        optimizer = torch.optim.SGD(bestmodel.parameters(), lr=0.001)
        from tqdm import tqdm
        for epoch in (range(epochs)):
            optimizer.zero_grad()
            #output = model(torch.cat([radar, torch.zeros(radar.shape[0], 1)], axis=1))
            output = bestmodel(radar)
            loss = F.mse_loss(output, buttons)
            loss.backward()
            optimizer.step()
            # p.set_description(f"Loss: {loss.item():2f} at epoch {epoch:2d}")
        
        state = bestmodel.state_dict()
        newmodel = MLP(*setup)
        newmodel.load_state_dict(state)
        return newmodel

#model = torch.load("model.pt")


# Create a car instance at the center of the screen
n_cars = 18*2
cars = [Car(SCREEN_WIDTH // 8 + 20, SCREEN_HEIGHT // 8 + i*0 + 20, 60, 30, model=MLP(*setup)) for i in range(n_cars)]

#import multiprocessing as mp

lastx, lasty = 0, 0

display = False
running = True

# Game loop
for j in range(100):
    for i in range(600 + j*100):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        screen.blit(track_image, (-lastx, -lasty))
            
        for car in cars:
            output = car.get_model_output(car.model)
            keys = pygame.key.get_pressed()


            car.accelerate()
            output += torch.rand(3) * 0.1
            if output.argmax(axis=-1) == 0:
                car.steer_left()
            elif output.argmax(axis=-1) == 2:
                car.steer_right()


            # Update the car
            car.update(track_mask)

            # Camera offset
            if not car.crashed:
                offset_x = car.x - SCREEN_WIDTH // 2
                offset_y = car.y - SCREEN_HEIGHT // 2
                output_backup = output

            car.draw(screen, lastx, lasty)
            if display:
                car.visualize(track_mask, screen, lastx, lasty, resolution=resolution)

            radar = car.get_radar_readings(track_mask, angle_range=45, resolution=resolution)
            speed = car.speed
            buttons = F.one_hot(output.argmax(), 3).float()
            car.recording.append(torch.cat([torch.tensor(radar), torch.tensor([speed]), buttons]))

        # draw output round2 on top left of the screen 
        # monospace font
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
        for epochs in [1, 5]:
            past.append((past_inter, epochs))
    past = past[::-1]

    for k in range(1):
        for i in range(n_cars):
            recording = torch.stack(cars[i].recording, axis=0)
            bestmodel = bestmodel.train(*[10, 3], recording)


    cars = [Car(SCREEN_WIDTH // 8 + 20, SCREEN_HEIGHT // 8 + 20 + torch.randint(0, 15, (1,)).item()*0, 60, 30, bestmodel.train(*past[i], torch.stack(cars[bestn].recording, axis=0)) if torch.rand(1).item() > -0.1 else bestmodel.train(*past[i], torch.stack(cars[i].recording, axis=0)))  for i in range(n_cars)]
    if running == False: break
pygame.quit()
