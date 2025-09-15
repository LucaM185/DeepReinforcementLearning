import numpy as np
import pygame
import math
import os
import torch
import sys
from services.Model import MLP


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
        self.original_x = x
        self.original_y = y
        self.x = x
        self.y = y
        self.angle = 0
        # Defaults tuned for general tracks; will be adapted per-track below
        self.speed = 2
        self.max_speed = 6
        self.min_speed = 1
        self.acceleration = 0.4
        self.deceleration = 0.2
        self.rotation_speed = 1.0
        # Adaptive parameters (set by `adapt_to_track`)
        self.radar_angle_range = 45
        self.radar_resolution = 8
        self._odometer = 0
        self.radar_recording = []
        self.speed_recording = []
        self.steer_recording = [] 
        self.longitudinal_recording = []
        self.model = model
        self.crashed = False

    def reset_position(self):
        self.x = self.original_x
        self.y = self.original_y
        self.angle = 0
        self.speed = 2
        self._odometer = 0
        self.crashed = False
        self.radar_recording = []
        self.steer_recording = []
        self.speed_recording = []
        self.longitudinal_recording = []

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

        # # Decelerate the car
        # if self.speed > 0:
        #     self.speed -= self.deceleration
        # elif self.speed < 0:
        #     self.speed += self.deceleration

        # Clamp the speed to the maximum speed
        self.speed = max(-self.max_speed, min(self.speed, self.max_speed))

        # Update the odometer
        self._odometer += self.speed

    def is_position_on_track(self, x, y, track_mask):
        # Check if the position is within the bounds of the track mask
        if x >= 0 and y >= 0 and x < track_mask.get_size()[0] and y < track_mask.get_size()[1]:
            # Get the alpha value of the track mask at the position
            return track_mask.get_at((int(x), int(y)))
        return False

    def accelerate(self):
        if self.crashed:
            return
        self.longitudinal_recording.append(0)
        self.speed = min(self.speed+self.acceleration, self.max_speed)
        self.speed_recording.append(self.speed)

    def brake(self):
        if self.crashed:
            return
        self.longitudinal_recording.append(2)
        self.speed = max(self.speed-self.deceleration, self.min_speed)        
        self.speed_recording.append(self.speed)

    def no_pedals(self):
        if self.crashed:
            return
        self.longitudinal_recording.append(1)
        self.speed_recording.append(self.speed)

    def steer_left(self):
        if self.crashed:
            return
        self.angle -= self.rotation_speed
        self.steer_recording.append(0)

    def go_straight(self):
        if self.crashed:
            return
        self.steer_recording.append(1)

    def steer_right(self):
        if self.crashed:
            return
        self.angle += self.rotation_speed
        self.steer_recording.append(2)

    def draw(self, surface, offset_x, offset_y):
        car_surface = pygame.Surface((self.original_width, self.original_height), pygame.SRCALPHA)
        car_surface.fill(RED)
        car_surface = pygame.transform.scale(car_surface, (self.width, self.height))
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)

        rect = rotated_car.get_rect(center=(self.x - offset_x, self.y - offset_y))
        surface.blit(rotated_car, rect.topleft)

    def get_radar_readings(self, track_mask, angle_range=45, resolution=8, save=False):
        # Use adaptive defaults if caller didn't pass specific values
        angle_range = angle_range or self.radar_angle_range
        resolution = resolution or self.radar_resolution
        readings = []
        if self.crashed:
            return [0] * (resolution+1)
        for i in np.arange(-angle_range, angle_range + 1, angle_range*2 / resolution):
            angle = self.angle + i
            distance = self.cast_ray(track_mask, angle)
            readings.append(distance)
        if save: self.radar_recording.append(readings)
        return readings

    def cast_ray(self, track_mask, angle):
        distance = 0
        x = self.x
        y = self.y
        while True:
            stepsize = 10
            x += math.cos(math.radians(angle))*stepsize
            y += math.sin(math.radians(angle))*stepsize
            distance += stepsize
            if not self.is_position_on_track(x, y, track_mask):
                backstepsize = 1
                while not self.is_position_on_track(x, y, track_mask):
                    x -= math.cos(math.radians(angle))*backstepsize
                    y -= math.sin(math.radians(angle))*backstepsize
                    distance -= backstepsize
                break
        return distance
    
    def get_model_output(self, mymodel: MLP, resolution: int, add_speed=True):
        # Get the radar readings
        readings = self.get_radar_readings(self.track_mask, resolution=resolution, save=True)
        readings = torch.tensor(readings).float()
        readings = readings.unsqueeze(0)
        if add_speed: readings = torch.cat([readings, torch.tensor([self.speed]).unsqueeze(0)], axis=-1)

        # Get the model output
        output = mymodel(readings)
        output = output.squeeze(0).detach()

        return output

    def get_observation(self, resolution: int):
        readings = self.get_radar_readings(self.track_mask, resolution=resolution, save=True)
        # Normalize distances (roughly) and speed
        max_dist = 500.0
        readings_norm = [min(max(r, 0.0), max_dist) / max_dist for r in readings]
        speed_norm = float(self.speed) / max(1.0, float(self.max_speed))
        obs = torch.tensor(readings_norm + [speed_norm], dtype=torch.float32)
        return obs

    def visualize(self, track_mask, surface, offset_x, offset_y, angle_range=80, resolution=8):
        readings = self.get_radar_readings(track_mask, angle_range, resolution)
        for i, reading in enumerate(readings):
            angle = self.angle + i / resolution * angle_range * 2 - angle_range
            end_x = self.x - offset_x + math.cos(math.radians(angle)) * reading
            end_y = self.y - offset_y + math.sin(math.radians(angle)) * reading
            pygame.draw.line(surface, (255, 0, 0), (self.x - offset_x, self.y - offset_y), (end_x, end_y), 1)

    @property
    def odometer(self):
        return self._odometer
        starting_point = (120, 120)
        return np.sqrt((self.x - starting_point[0])**2 + (self.y - starting_point[1])**2)

    @property
    def steering_tensor(self):
        return torch.nn.functional.one_hot(torch.tensor(self.steer_recording).long(), 3).float()
    
    @property
    def longitudinal_tensor(self):
        return torch.nn.functional.one_hot(torch.tensor(self.longitudinal_recording).long(), 3).float()

    @property
    def action_tensor(self):
        steering = self.steering_tensor
        longitudinal = self.longitudinal_tensor
        # Align lengths to avoid mismatch when one stops updating (e.g., after crash)
        min_len = min(steering.shape[0], longitudinal.shape[0]) if steering.ndim > 0 and longitudinal.ndim > 0 else 0
        if min_len == 0:
            return torch.zeros((0, 6)).float()
        return torch.cat([steering[:min_len], longitudinal[:min_len]], axis=-1)
    
    @property
    def speed_tensor(self):
        return torch.tensor(self.speed_recording).float()

    @property
    def radar_tensor(self):
        return torch.tensor(self.radar_recording).float()
    
    @property
    def environment_tensor(self):
        radar = self.radar_tensor
        speed = self.speed_tensor
        # Ensure same timesteps length
        min_len = min(radar.shape[0], speed.shape[0]) if radar.ndim > 0 and speed.ndim > 0 else 0
        if min_len == 0:
            return torch.zeros((0, radar.shape[1] + 1 if radar.ndim == 2 else 1)).float()
        return torch.cat([radar[:min_len], speed[:min_len].unsqueeze(-1)], axis=-1)