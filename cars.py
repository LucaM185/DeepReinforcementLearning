import numpy as np
import pygame
import math
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Car Steering on Track with Camera')

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Clock to control the frame rate
clock = pygame.time.Clock()

# Load and scale the track image
try:
    track_image = pygame.image.load("monza.png").convert_alpha()
    track_image = pygame.transform.scale(track_image, (track_image.get_width() * 2, track_image.get_height() * 2))
    track_rect = track_image.get_rect()
except pygame.error:
    print("Failed to load the image 'monza.png'. Please make sure the file exists in the same directory as this script.")
    pygame.quit()
    sys.exit()

track_mask = pygame.mask.from_surface(track_image)

# radar parameters
resolution = 8

class Car:
    def __init__(self, x, y, width, height, model):
        self.original_width = width
        self.original_height = height
        self.width = width // 2  # Make the car smaller
        self.height = height // 2
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.max_speed = 2
        self.acceleration = 0.2
        self.deceleration = 0.1
        self.rotation_speed = 0.6
        self.odometer = 0
        self.recording = []
        self.model = model

    def update(self, track_mask):
        # Calculate new position based on the speed and angle
        new_x = self.x + math.cos(math.radians(self.angle)) * self.speed
        new_y = self.y + math.sin(math.radians(self.angle)) * self.speed

        # Check if the new position is on the track
        if self.is_position_on_track(new_x, new_y, track_mask):
            # Update the position
            self.x = new_x
            self.y = new_y
        else:
            # Stop the car if it's not on the track
            self.speed = 0

        # Decelerate the car
        if self.speed > 0:
            self.speed -= self.deceleration
        elif self.speed < 0:
            self.speed += self.deceleration

        # Clamp the speed to the maximum speed
        self.speed = max(-self.max_speed, min(self.speed, self.max_speed))

        # Update the odometer
        self.odometer += self.speed

    def is_position_on_track(self, x, y, track_mask):
        # Check if the position is within the bounds of the track mask
        if x >= 0 and y >= 0 and x < track_mask.get_size()[0] and y < track_mask.get_size()[1]:
            # Get the alpha value of the track mask at the position
            return track_mask.get_at((int(x), int(y)))
        return False

    def accelerate(self):
        self.speed += self.acceleration

    def brake(self):
        if self.speed > 0:
            self.speed -= self.acceleration
        else:
            self.speed -= self.deceleration

    def steer_left(self):
        self.angle -= self.rotation_speed

    def steer_right(self):
        self.angle += self.rotation_speed

    def draw(self, surface, offset_x, offset_y):
        # Create a new surface with car dimensions
        car_surface = pygame.Surface((self.original_width, self.original_height), pygame.SRCALPHA)
        # Fill the surface with the car color
        car_surface.fill(RED)
        # Scale the car down
        car_surface = pygame.transform.scale(car_surface, (self.width, self.height))
        # Rotate the car surface to match the current angle
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        # Get the new rect and blit to the screen at the center of the car
        rect = rotated_car.get_rect(center=(self.x - offset_x, self.y - offset_y))
        surface.blit(rotated_car, rect.topleft)

    def get_radar_readings(self, track_mask, angle_range=45, resolution=8):
        readings = []
        for i in np.arange(-angle_range, angle_range + 1, angle_range*2 / resolution):
            angle = self.angle + i
            distance = self.cast_ray(track_mask, angle)
            readings.append(distance)
        return readings

    def cast_ray(self, track_mask, angle):
        distance = 0
        x = self.x
        y = self.y
        while True:
            x += math.cos(math.radians(angle))
            y += math.sin(math.radians(angle))
            distance += 1
            if not self.is_position_on_track(x, y, track_mask):
                break
        return distance
    
    def get_model_output(self, mymodel):
        # Get the radar readings
        readings = self.get_radar_readings(track_mask)
        readings = torch.tensor(readings).float()
        readings = readings.unsqueeze(0)

        # Get the model output
        output = mymodel(readings)
        output = output.squeeze(0).detach().numpy()

        return output

    def visualize(self, track_mask, surface, offset_x, offset_y, angle_range=45, resolution=8):
        readings = self.get_radar_readings(track_mask, angle_range, resolution)
        for i, reading in enumerate(readings):
            angle = self.angle + i / resolution * angle_range * 2 - angle_range
            end_x = self.x - offset_x + math.cos(math.radians(angle)) * reading
            end_y = self.y - offset_y + math.sin(math.radians(angle)) * reading
            pygame.draw.line(surface, (255, 0, 0), (self.x - offset_x, self.y - offset_y), (end_x, end_y), 1)

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
        return self.fc2(x)

#model = torch.load("model.pt")


# Create a car instance at the center of the screen
n_cars = 3
cars = [Car(SCREEN_WIDTH // 8 + 20, SCREEN_HEIGHT // 8 + 20, 60, 30, model=MLP(9, 4, 64, 3)) for _ in range(n_cars)]

#import multiprocessing as mp

lastx, lasty = 0, 0

display = True
running = True

# Game loop
for j in range(100):
    for i in range(500):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        screen.blit(track_image, (-lastx, -lasty))
            
        for car in cars:
            output = car.get_model_output(car.model)
            keys = pygame.key.get_pressed()


            if True:
                if output[0] > 0.03:
                    car.accelerate()
                if output[1] > 0.3:  # steer 
                    if output[2] > output[3]: # left
                        car.steer_left()
                    if output[3] > output[2]: # right
                        car.steer_right()


            # Update the car
            car.update(track_mask)

            # Camera offset
            offset_x = car.x - SCREEN_WIDTH // 2
            offset_y = car.y - SCREEN_HEIGHT // 2

            if display:
                car.draw(screen, lastx, lasty)
                car.visualize(track_mask, screen, lastx, lasty, resolution=resolution)

            radar = car.get_radar_readings(track_mask, angle_range=45, resolution=resolution)
            speed = car.speed
            buttons = torch.tensor([output[0] > 0.1, output[1] > 0.3, output[2] > output[3], output[2] < output[3]], dtype=torch.float32)
            car.recording.append(torch.cat([torch.tensor(radar), torch.tensor([speed]), buttons]))

        lastx, lasty = offset_x, offset_y
        pygame.display.flip()
        clock.tick(1000)
    
        if keys[pygame.K_q]:
            running = False
            break

    bestOdometer = -10
    for n, car in enumerate(cars):
        if car.odometer > bestOdometer:
            bestOdometer = car.odometer
            bestmodel = car.model
            bestn = n
    print(n)
    recording = torch.stack(cars[bestn].recording, axis=0)
    car = cars[bestn]

    # tune model
    past_interaction = 10
    radar = recording[:-past_interaction, :9]
    buttons = (recording[:-past_interaction, 10:] + recording[past_interaction:, 10:])/2
    speed = recording[past_interaction:, 9]

    # make labels so that buttons that lead to high speed 20 steps after are rewarded
    speedthresh = 1.5
    buttons[speed < speedthresh] = (buttons[speed < speedthresh] < 0.5) + 0.0
    buttons = buttons[speed < speedthresh]
    radar = radar[speed < speedthresh]

    # train model   
    optimizer = torch.optim.SGD(bestmodel.parameters(), lr=0.0003)
    from tqdm import tqdm
    for epoch in (range(5)):
        optimizer.zero_grad()
        #output = model(torch.cat([radar, torch.zeros(radar.shape[0], 1)], axis=1))
        output = bestmodel(radar)
        loss = F.mse_loss(output, buttons)
        loss.backward()
        optimizer.step()
        # p.set_description(f"Loss: {loss.item():2f} at epoch {epoch:2d}")
    
    print(f"{car.odometer:2f} - {speed.mean():2f} -{car}")


    cars = [Car(SCREEN_WIDTH // 8 + 20, SCREEN_HEIGHT // 8 + 20 + i * 10, 60, 30, car.model) for i in range(n_cars)]
    if running == False: break
pygame.quit()
