import pygame
import torch

from services.Track import * 
from services.Car import Car
from services.Model import MLP
from services.Dataset import MyDataset

torch.manual_seed(42)

pygame.init()



SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (0, 0, 0)
RED = (255, 0, 0)

#track_image, track_mask = get_track_mask(r"monza.png")
#starting_point = (120, 120)
track_image, track_mask, starting_point = make_new_track()
starting_point = starting_point[0]*8, starting_point[1]*8


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Car Steering on Track with Camera')
clock = pygame.time.Clock()

resolution = 24

n_cars = 30
def analyze_track_complexity(mask):
    # crude metric: average radar span from center across many angles
    # higher value -> more open (straights); lower -> more twisty
    w, h = mask.get_size()
    cx, cy = w//2, h//2
    import math
    samples = 16
    total = 0
    for a in range(samples):
        ang = a/samples*360
        # cast a short ray outward to estimate openness
        dist = 0
        x, y = cx, cy
        while mask.get_at((int(x), int(y))):
            x += math.cos(math.radians(ang))*5
            y += math.sin(math.radians(ang))*5
            dist += 5
            if dist > max(w, h):
                break
        total += dist
    avg = total / samples
    # Normalize roughly to [0,1]
    norm = max(0.0, min(1.0, (avg - 200) / 400))
    return norm

track_complexity = analyze_track_complexity(track_mask)

# Create cars with adaptive radar and dynamics based on track_complexity
cars = []
for i in range(n_cars):
    model = MLP(*MLP.setup, lr=0.003)
    # Slightly perturb initial weights so each car starts differently
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    car = Car(starting_point[0], + starting_point[1], 60, 30, model=model, track_mask=track_mask)
    # Track complexity 0 -> twisty, 1 -> open straights
    openness = track_complexity
    # Adapt radar: twisty -> wider angle, higher resolution; straight -> narrow angle, longer range
    car.radar_angle_range = int(60 - openness*30)  # [60 -> 30]
    car.radar_resolution = int(8 + openness*8)     # [8 -> 16]
    # Adapt speed: straights -> higher max_speed, twisty -> lower max_speed, higher rotation
    car.max_speed = 4 + int(openness*8)            # [4 -> 12]
    car.acceleration = 0.3 + openness*0.5
    car.deceleration = 0.2 + (1-openness)*0.3
    car.rotation_speed = 1.2 - openness*0.6
    cars.append(car)
dataset = MyDataset()

lastx, lasty = 0, 0

display_radar = True
display = True
running = True
manual_control = False

backupmodel = cars[0].model
backupodometer = 0

# Game loop
# Increased limits: outer loop (episodes) and inner loop (steps per episode)
MAX_EPISODES = 10000
STEPS_PER_EPISODE_BASE = 5000

for j in range(MAX_EPISODES):
    for i in range(STEPS_PER_EPISODE_BASE + j * 500):
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

        # Focus camera and HUD on the alive car with the highest distance driven
        alive_indices = [i for i, c in enumerate(cars) if not c.crashed]
        target_indices = alive_indices if alive_indices else list(range(len(cars)))
        best_idx = max(target_indices, key=lambda k: cars[k].odometer)
        focused_car = cars[best_idx]
        offset_x = focused_car.x - SCREEN_WIDTH // 2
        offset_y = focused_car.y - SCREEN_HEIGHT // 2
        output_backup = outputs[best_idx]
        speed = focused_car.speed

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

    # Gradient step on best run, then distribute gradient-updated copies with varying LRs
    base_model = backupmodel.get_model_copy()
    best_car = cars[bestn]
    if best_car.environment_tensor.shape[0] > 0 and best_car.action_tensor.shape[0] > 0:
        base_model = base_model.train(best_car.environment_tensor, best_car.action_tensor, 240)

    for i, car in enumerate(cars):
        newmodel = base_model.get_model_copy()
        # Assign different learning rates across the fleet (low -> high)
        span = max(1, n_cars - 1)
        frac = i / span
        lr_scale = 0.2 + 0.8 * frac  # [0.2x .. 1.0x]
        newmodel.lr = base_model.lr * lr_scale
        if best_car.environment_tensor.shape[0] > 0 and best_car.action_tensor.shape[0] > 0:
            newmodel = newmodel.train(best_car.environment_tensor, best_car.action_tensor, 240)
        car.model = newmodel
    for car in cars: car.reset_position()
    if running == False: break

pygame.quit()
