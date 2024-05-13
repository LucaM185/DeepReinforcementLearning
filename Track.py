import pygame
import sys

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

def get_track_mask(name=r"C:\Users\lucam\Desktop\VScode\Cars\monza.png"):
    # Load and scale the track image

    track_image = pygame.image.load(name).convert_alpha()
    track_image = pygame.transform.scale(track_image, (track_image.get_width() * 2, track_image.get_height() * 2))
    track_rect = track_image.get_rect()

    track_mask = pygame.mask.from_surface(track_image)
    return track_image, track_mask