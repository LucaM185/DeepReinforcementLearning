import pygame
import torch

from services.Track import * 
from services.Car import Car
from services.Model import ActorCritic, PPOTrainer
from services.Dataset import RolloutBuffer

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

n_cars = 10
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
    car = Car(starting_point[0], + starting_point[1], 60, 30, model=None, track_mask=track_mask)
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

lastx, lasty = 0, 0

display_radar = True
display = True
running = True
manual_control = False
with torch.no_grad():
    obs_dim = int(cars[0].get_observation(resolution).shape[0])
actor_critic = ActorCritic(obs_dim, hidden_size=128, n_layers=2)
trainer = PPOTrainer(actor_critic, lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_coef=0.3, update_epochs=6, minibatch_size=128, entropy_coef=0.02)

ROLLOUT_STEPS = 256
MAX_UPDATES = 10000

for update_idx in range(MAX_UPDATES):
    buffer = RolloutBuffer(ROLLOUT_STEPS, n_cars, obs_dim)
    with torch.no_grad():
        obs = torch.stack([car.get_observation(resolution) for car in cars], dim=0)
    all_dead = False
    for t in range(ROLLOUT_STEPS):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        screen.fill(WHITE)
        screen.blit(track_image, (-lastx, -lasty))
        with torch.no_grad():
            actions, logprobs, entropy, values = actor_critic.get_action_and_value(obs)
        # Step environments
        for i, car in enumerate(cars):
            lat = int(actions[i, 0].item())
            lon = int(actions[i, 1].item())
            if lat == 0:
                car.steer_left()
            elif lat == 2:
                car.steer_right()
            else:
                car.go_straight()
            if lon == 0:
                car.accelerate()
            elif lon == 2:
                car.brake()
            else:
                car.no_pedals()
            car.update(track_mask)
        # Compute rewards and dones (encourage distance progress, penalize crash)
        rewards = torch.zeros(n_cars, dtype=torch.float32)
        dones = torch.zeros(n_cars, dtype=torch.float32)
        for i, car in enumerate(cars):
            if car.crashed:
                rewards[i] = -10.0
                dones[i] = 1.0
            else:
                # Reward forward motion via speed; small living bonus
                rewards[i] = float(car.speed) * 0.2 + 0.05
        # Draw
        if display:
            for car in cars:
                car.draw(screen, lastx, lasty)
        if display_radar:
            for car in cars:
                car.visualize(track_mask, screen, lastx, lasty, resolution=resolution)
        # Focus camera on best alive
        alive_indices = [i for i, c in enumerate(cars) if not c.crashed]
        target_indices = alive_indices if alive_indices else list(range(len(cars)))
        best_idx = max(target_indices, key=lambda k: cars[k].odometer)
        focused_car = cars[best_idx]
        offset_x = focused_car.x - SCREEN_WIDTH // 2
        offset_y = focused_car.y - SCREEN_HEIGHT // 2
        lastx, lasty = offset_x, offset_y
        if display:
            font = pygame.font.Font(None, 36)
            mystring = f"Lat: {int(actions[best_idx,0].item())}\nLong: {int(actions[best_idx,1].item())}\nSpd: {focused_car.speed:.2f}"
            for n, line in enumerate(mystring.split("\n")):
                text = font.render(line, True, (0, 255, 0))
                screen.blit(text, (10, 10 + n*36))
        pygame.display.flip()
        clock.tick(2000)
        # Store
        buffer.add(obs, actions, logprobs, rewards, dones, values)
        # Check if all dead; if so, stop collecting and do update
        if int(dones.sum().item()) == n_cars:
            all_dead = True
            break
        with torch.no_grad():
            obs = torch.stack([car.get_observation(resolution) for car in cars], dim=0)
        if keys[pygame.K_q] or not running:
            running = False
            break
    if not running:
        break
    with torch.no_grad():
        if all_dead:
            last_values = torch.zeros(n_cars, dtype=torch.float32)
        else:
            _, _, last_values = actor_critic.forward(obs)
    buffer.compute_gae(last_values, gamma=trainer.gamma, gae_lambda=trainer.gae_lambda)
    stats = trainer.update(buffer)
    print(f"Update {update_idx} | Policy {stats['policy_loss']:.4f} | Value {stats['value_loss']:.4f} | Ent {stats['entropy']:.4f}")
    # Reset all cars after each update
    for car in cars:
        car.reset_position()
pygame.quit()
