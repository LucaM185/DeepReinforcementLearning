import pygame
import torch
import os
import sys
import subprocess
import tempfile
import json

from services.Track import get_track_mask, make_new_track, SCALE_FACTOR
from services.Car import Car
from services.Model import ActorCritic, PPOTrainer
from services.Dataset import RolloutBuffer


torch.manual_seed(42)
# Exposed run parameters

SHOW_FPS = int(os.environ.get("SHOW_FPS", "60"))


pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (0, 0, 0)
RED = (255, 0, 0)
# Exposed RL and sim parameters
PIXEL_REWARD_SCALE = 1.6
RESOLUTION = 40
N_CARS = 10
ROLLOUT_STEPS = 256
MAX_UPDATES = 10000

# Keep track of the number of training runs
run_number = 0

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Car Steering on Track with Camera')
clock = pygame.time.Clock()

# track_image, track_mask = get_track_mask(r"monza.png")
# starting_point = (120, 120)
track_image, track_mask, starting_point = make_new_track()
starting_point = starting_point[0]*SCALE_FACTOR*2, starting_point[1]*SCALE_FACTOR*2


# Bias injection parameters (change these to tune behavior)
# BIAS_PERIOD_STEPS: every N steps start a bias window
# BIAS_DURATION_STEPS: how many steps the bias lasts inside each window
# BIAS_MAGNITUDE: added logit for left/right during bias window
BIAS_PERIOD_STEPS = 450
BIAS_DURATION_STEPS = 30
BIAS_MAGNITUDE = 0.3
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
for i in range(N_CARS):
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
    obs_dim = int(cars[0].get_observation(RESOLUTION).shape[0])
actor_critic = ActorCritic(obs_dim, hidden_size=128, n_layers=2)
trainer = PPOTrainer(actor_critic, lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_coef=0.3, update_epochs=6, minibatch_size=128, entropy_coef=0.02)

# Create a directory for models if it doesn't exist
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

def save_model(path):
    torch.save(actor_critic.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path):
    if os.path.exists(path):
        actor_critic.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}")

for update_idx in range(MAX_UPDATES):
    # Every 10th run, show in real-time, otherwise run at full speed
    display = True
    SHOW_FPS = 60 # Real-time FPS

    buffer = RolloutBuffer(ROLLOUT_STEPS, N_CARS, obs_dim)
    with torch.no_grad():
        obs = torch.stack([car.get_observation(RESOLUTION) for car in cars], dim=0)
    all_dead = False
    for t in range(ROLLOUT_STEPS):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_model(BEST_MODEL_PATH)
                if event.key == pygame.K_l:
                    load_model(BEST_MODEL_PATH)

        keys = pygame.key.get_pressed()
        if display:
            screen.fill(WHITE)
            screen.blit(track_image, (-lastx, -lasty))
        # Bias schedule: every BIAS_PERIOD_STEPS sim steps, apply BIAS_DURATION_STEPS of lateral bias
        # Determine current global step within this update
        global_step = t
        lateral_bias = None
        if (global_step % BIAS_PERIOD_STEPS) < BIAS_DURATION_STEPS:
            # Choose direction deterministically per window using RNG seeded by update+window
            window_idx = global_step // BIAS_PERIOD_STEPS
            rng_seed = (update_idx * 1000003 + window_idx)
            torch.manual_seed(rng_seed)
            direction = torch.randint(0, 2, (1,)).item()  # 0 -> left, 1 -> right
            bias_vector = torch.zeros(3)
            # logits order assumed [left, straight, right]
            if direction == 0:
                bias_vector[0] = BIAS_MAGNITUDE
            else:
                bias_vector[2] = BIAS_MAGNITUDE
            # broadcast to all envs
            lateral_bias = bias_vector.unsqueeze(0).expand(obs.shape[0], -1).clone()
        with torch.no_grad():
            actions, logprobs, entropy, values = actor_critic.get_action_and_value(obs, lateral_logits_bias=lateral_bias)
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
        rewards = torch.zeros(N_CARS, dtype=torch.float32)
        dones = torch.zeros(N_CARS, dtype=torch.float32)
        # We'll also compute and store per-source reward components for debugging/printing
        speed_rewards = [0.0] * N_CARS
        pixel_rewards = [0.0] * N_CARS
        crash_rewards = [0.0] * N_CARS
        for i, car in enumerate(cars):
            if car.crashed:
                # Determine the last recorded speed before the crash (fallback to current speed)
                try:
                    prior_speed = float(car.speed_recording[-1]) if len(car.speed_recording) > 0 else float(car.speed)
                except Exception:
                    prior_speed = float(car.speed)
                # Base crash penalty plus an additional penalty that grows with speed^2
                # This strongly discourages crashing at high speed.
                crash_penalty = -10.0
                FAST_CRASH_PENALTY_MULTIPLIER = 0.5
                crash_rewards[i] = float(crash_penalty - FAST_CRASH_PENALTY_MULTIPLIER * (prior_speed ** 2))
                rewards[i] = crash_rewards[i]
                dones[i] = 1.0
            else:
                # Reward forward motion via speed; small living bonus
                speed_val = float(car.speed) * 0.2 + 0.05
                speed_rewards[i] = float(speed_val)
                pixel_val = 0.0
                # Sum components
                rewards[i] = float(speed_rewards[i] + pixel_rewards[i])
        # Aggregate per-source rewards across all cars and print a compact summary
        try:
            total_crash = sum(crash_rewards)
            total_speed = sum(speed_rewards)
            total_pixel = sum(pixel_rewards)
            total_reward = float(rewards.sum().item())
            # Compact single-line print: Crash, Speed, Pixel, Total
            print(f"Crash {total_crash:+.4f} | Speed {total_speed:+.4f} | Pixel {total_pixel:+.4f} | Total {total_reward:+.4f}")
        except Exception:
            pass
        # Draw
        if display:
            for car in cars:
                car.draw(screen, lastx, lasty)
        if display_radar:
            for car in cars:
                car.visualize(track_mask, screen, lastx, lasty, resolution=RESOLUTION)
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
            # Global simulation step number
            sim_step = update_idx * ROLLOUT_STEPS + t
            step_text = f"Step: {sim_step}"
            step_surf = font.render(step_text, True, (0, 255, 0))
            screen.blit(step_surf, (10, 10))

            mystring = f"Lat: {int(actions[best_idx,0].item())}\nLong: {int(actions[best_idx,1].item())}\nSpd: {focused_car.speed:.2f}"
            for n, line in enumerate(mystring.split("\n")):
                text = font.render(line, True, (0, 255, 0))
                # draw the other status slightly lower so it doesn't overlap the step counter
                screen.blit(text, (10, 50 + n*36))
            pygame.display.flip()
            clock.tick(SHOW_FPS)
        # Write lightweight state for the blur-follow helper (best-effort)

        # Store
        buffer.add(obs, actions, logprobs, rewards, dones, values)
        # Check if all dead; if so, stop collecting and do update
        if int(dones.sum().item()) == N_CARS:
            all_dead = True
            break
        with torch.no_grad():
            obs = torch.stack([car.get_observation(RESOLUTION) for car in cars], dim=0)
        if keys[pygame.K_q] or not running:
            running = False
            break
    if not running:
        break
    with torch.no_grad():
        if all_dead:
            last_values = torch.zeros(N_CARS, dtype=torch.float32)
        else:
            _, _, last_values = actor_critic.forward(obs)
    buffer.compute_gae(last_values, gamma=trainer.gamma, gae_lambda=trainer.gae_lambda)
    stats = trainer.update(buffer)
    print(f"Update {update_idx} | Policy {stats['policy_loss']:.4f} | Value {stats['value_loss']:.4f} | Ent {stats['entropy']:.4f}")
    # Reset cars infrequently: only if all are dead or every 500 updates
    if all_dead or ((update_idx + 1) % 500 == 0):
        for car in cars:
            car.reset_position()
        run_number += 1
pygame.quit()
