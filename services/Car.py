import numpy as np
import pygame
import math
import torch


# Tunable defaults (easy to tweak)
DEFAULT_CAR_SPEED = 2
DEFAULT_MAX_SPEED = 6
DEFAULT_MIN_SPEED = 1
DEFAULT_ACCELERATION = 0.4
DEFAULT_DECELERATION = 0.2
DEFAULT_ROTATION_SPEED = 1.0
DEFAULT_RADAR_ANGLE_RANGE = 85
DEFAULT_RADAR_RESOLUTION = 16
DEFAULT_MAX_RADAR_DISTANCE = 500.0  # for observation normalization

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
        self.speed = DEFAULT_CAR_SPEED
        self.max_speed = DEFAULT_MAX_SPEED
        self.min_speed = DEFAULT_MIN_SPEED
        self.acceleration = DEFAULT_ACCELERATION
        self.deceleration = DEFAULT_DECELERATION
        self.rotation_speed = DEFAULT_ROTATION_SPEED
        # Adaptive parameters (set by `adapt_to_track`)
        self.radar_angle_range = DEFAULT_RADAR_ANGLE_RANGE
        self.radar_resolution = DEFAULT_RADAR_RESOLUTION
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
        self.speed = DEFAULT_CAR_SPEED
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

    def get_radar_readings(self, track_mask, angle_range=None, resolution=None, save=False):
        # Use adaptive defaults if caller didn't pass specific values
        angle_range = self.radar_angle_range if angle_range is None else angle_range
        resolution = self.radar_resolution if resolution is None else resolution
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
    
    def get_observation(self, resolution: int):
        readings = self.get_radar_readings(self.track_mask, resolution=resolution, save=True)
        # Normalize distances (roughly) and speed
        max_dist = DEFAULT_MAX_RADAR_DISTANCE
        readings_norm = [min(max(r, 0.0), max_dist) / max_dist for r in readings]
        speed_norm = float(self.speed) / max(1.0, float(self.max_speed))
        obs = torch.tensor(readings_norm + [speed_norm], dtype=torch.float32)
        return obs

    def visualize(self, track_mask, surface, offset_x, offset_y, angle_range=120, resolution=16):
        readings = self.get_radar_readings(track_mask, angle_range, resolution)
        for i, reading in enumerate(readings):
            angle = self.angle + i / resolution * angle_range * 2 - angle_range
            end_x = self.x - offset_x + math.cos(math.radians(angle)) * reading
            end_y = self.y - offset_y + math.sin(math.radians(angle)) * reading
            pygame.draw.line(surface, (255, 0, 0), (self.x - offset_x, self.y - offset_y), (end_x, end_y), 1)

    @property
    def odometer(self):
        return self._odometer