import pygame
import sys

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


def get_track_mask(name=r"C:\Users\lucam\Desktop\VScode\Cars\monza.png"):
    # Initialize Pygame
    pygame.init()

    # Load and scale the track image
    track_image = pygame.image.load(name).convert_alpha()
    track_image = pygame.transform.scale(track_image, (track_image.get_width() * 2, track_image.get_height() * 2))

    # Invert black and white in the image
    track_array = pygame.surfarray.pixels3d(track_image)
    track_array[:] = 255 - track_array

    # Create a mask from the modified surface
    track_mask = pygame.mask.from_surface(track_image)
    return track_image, track_mask

import numpy as np
import cv2

image = np.zeros((600, 800), np.uint8)
prev = None
mousedown = False
starting_point = None

def draw(event, x, y, flags, param):
    global image
    global prev
    global mousedown
    global starting_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if prev == None:
            prev = (x, y)
        mousedown = True
        if starting_point == None:
            starting_point = (x, y)

    if mousedown:
        cv2.line(image, prev, (x, y), (0, 0, 0, 0), 10)
        prev = (x, y)

    if event == cv2.EVENT_LBUTTONUP:
        mousedown = False
    

def make_new_track():
    global image
    image = np.zeros((600, 800, 4), np.uint8) + 255
    while True:
        cv2.imshow("Track", image)
        # mouse callback
        cv2.setMouseCallback("Track", draw)
        k = cv2.waitKey(10)
    
        if k == 27:
            break
    
    image = cv2.resize(image, (800*4, 600*4))
    image[:,:,3] = 255-image[:,:,3]
    cv2.imwrite("track.png", image)
    cv2.destroyAllWindows()

    return *get_track_mask("track.png"), starting_point 

    