import pygame
import torch

from Track import * 
from Car import Car
from Model import MLP
from Dataset import MyDataset

torch.manual_seed(42)

pygame.init()



SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (0, 0, 0)
RED = (255, 0, 0)

track_image, track_mask = get_track_mask(r"monza.png")
starting_point = (120, 120)
#track_image, track_mask, starting_point = make_new_track()
#starting_point = starting_point[0]*8, starting_point[1]*8


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Car Steering on Track with Camera')
clock = pygame.time.Clock()

resolution = 24

n_cars = 1
cars = [Car(starting_point[0], + starting_point[1], 60, 30, model=MLP(*MLP.setup, lr=0.003), track_mask=track_mask) for i in range(n_cars)]
dataset = MyDataset()

lastx, lasty = 0, 0

display_radar = True
display = True
running = True
manual_control = False

backupmodel = cars[0].model
backupodometer = 0

# Game loop
for j in range(1000):
    for i in range(1000 + j * 100):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()

        screen.fill(WHITE)
        screen.blit(track_image, (-lastx, -lasty))
        
        outputs = [car.get_model_output(car.model, resolution) for car in cars]
            
        for n, (car, output) in enumerate(zip(cars, outputs)):
            lateral, longitudinal = output[:3], output[3:] 

            if manual_control and n == n_cars-1:
                if keys[pygame.K_LEFT]:
                    car.steer_left()
                elif keys[pygame.K_RIGHT]:
                    car.steer_right()
                else:
                    car.go_straight()
                if keys[pygame.K_UP]:
                    car.accelerate()
                elif keys[pygame.K_DOWN]:
                    car.brake()
            else:
                if lateral.argmax(axis=-1).item() == 0:
                    car.steer_left()
                elif lateral.argmax(axis=-1).item() == 2:
                    car.steer_right()
                else:
                    car.go_straight()
                
                if longitudinal.argmax(axis=-1).item() == 0:
                    car.accelerate()
                elif longitudinal.argmax(axis=-1).item() == 2:
                    car.brake()
                else:
                    car.no_pedals()
                
            car.update(track_mask)

            if not car.crashed:
                offset_x = car.x - SCREEN_WIDTH // 2
                offset_y = car.y - SCREEN_HEIGHT // 2
                output_backup = output

            if display: car.draw(screen, lastx, lasty)
            if display_radar: car.visualize(track_mask, screen, lastx, lasty, resolution=resolution)

            # radar = car.get_radar_readings(track_mask, angle_range=45, resolution=resolution)
            # speed = car.speed
            speed = car.speed

        if display:
            font = pygame.font.Font(None, 36)
            choicelateral = output_backup[:3].argmax(axis=-1).item()
            choicelongitudinal = output_backup[3:].argmax(axis=-1).item()
            mystring = f"Lat: {choicelateral} @ {output_backup[choicelateral]:.2f} \nLong {choicelongitudinal} @ {output_backup[choicelongitudinal+3]:.2f} \nSpd: {speed:.2f}"
            for n, line in enumerate(mystring.split("\n")):
                text = font.render(line, True, (0, 255, 0))
                screen.blit(text, (10, 10 + n*36))

        lastx, lasty = offset_x, offset_y
        pygame.display.flip()
        clock.tick(2000)
    
        if keys[pygame.K_q]:
            running = False
            break
        
        if [car.crashed for car in cars].count(True) == n_cars:
            break

    bestOdometer = -10
    for n, car in enumerate(cars):
        if car.odometer > bestOdometer:
            bestOdometer = car.odometer
            bestmodel = car.model
            bestn = n

    if bestOdometer > backupodometer:
        backupmodel = cars[bestn].model.get_model_copy()
        backupodometer = bestOdometer
        print(f"Backup model updated, odo {bestOdometer:.2f}, lr {backupmodel.lr:.6f}")
    else:
        print(f"{cars[bestn].odometer:2f}")

    past = torch.randint(1, 100, (n_cars,))
    past = [240]

    for n, car in enumerate(cars): car.model = bestmodel.train(car.environment_tensor, car.action_tensor, past[n])
    for car in cars: car.reset_position()
    if running == False: break

    if bestOdometer < backupodometer and torch.randint(0, 100, (1,)) > 95: 
        cars[bestn].model = backupmodel.get_model_copy()
        backupmodel.lr *= 0.8
        print(f"Backup model loaded, lr {backupmodel.lr:.6f}")

pygame.quit()
