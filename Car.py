import numpy as np
import pygame
import math
import os
import torch
import sys


# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class Car:
    def __init__(self, x, y, width, height, model, track_mask):
        self.track_mask = track_mask
        self.original_width = width
        self.original_height = height
        self.width = width // 2  # Make the car smaller
        self.height = height // 2
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 2
        self.max_speed = 3
        self.acceleration = 0.2
        self.deceleration = 0.1
        self.rotation_speed = 1.0
        self.odometer = 0
        self.recording = []
        self.model = model
        self.crashed = False

    def update(self, track_mask):
        # Calculate new position based on the speed and angle
        new_x = self.x + math.cos(math.radians(self.angle)) * self.speed
        new_y = self.y + math.sin(math.radians(self.angle)) * self.speed

        # Check if the new position is on the track
        if self.is_position_on_track(new_x, new_y, track_mask) and not self.crashed:
            # Update the position
            self.x = new_x
            self.y = new_y
        else:
            # Stop the car if it's not on the track
            self.speed = 0
            self.crashed = True

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
        if self.crashed:
            return [0] * (resolution+1)
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
    
    def get_model_output(self, mymodel, resolution):
        # Get the radar readings
        readings = self.get_radar_readings(self.track_mask, resolution=resolution)
        readings = torch.tensor(readings).float()
        readings = readings.unsqueeze(0)

        # Get the model output
        output = mymodel(readings)
        output = output.squeeze(0).detach()

        return output

    def visualize(self, track_mask, surface, offset_x, offset_y, angle_range=45, resolution=8):
        readings = self.get_radar_readings(track_mask, angle_range, resolution)
        for i, reading in enumerate(readings):
            angle = self.angle + i / resolution * angle_range * 2 - angle_range
            end_x = self.x - offset_x + math.cos(math.radians(angle)) * reading
            end_y = self.y - offset_y + math.sin(math.radians(angle)) * reading
            pygame.draw.line(surface, (255, 0, 0), (self.x - offset_x, self.y - offset_y), (end_x, end_y), 1)
