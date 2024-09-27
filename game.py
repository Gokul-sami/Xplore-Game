import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp
import random
import time

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))  # Windowed mode
pygame.display.set_caption("Dragon Game")
width, height = screen.get_size()  # Get current window size

# Load images
dragon_image = pygame.image.load('assets/dragon-removebg.png').convert_alpha()  # Your dragon image
egg_images = [
    pygame.image.load('assets/egg2.png').convert_alpha(),
    pygame.image.load('assets/egg3.png').convert_alpha(),
    pygame.image.load('assets/egg4.png').convert_alpha()
]
fireball_image = pygame.image.load('assets/fb1.gif').convert_alpha()  # Your fireball image
# Removed background_image loading since we're keeping a white background
# background_image = pygame.image.load('assets/background.jpg').convert()  # Your background image

# Scale images
def scale_image(image, scale_factor):
    img_width, img_height = image.get_size()
    scaled_width = int(img_width * scale_factor)
    scaled_height = int(img_height * scale_factor)
    return pygame.transform.scale(image, (scaled_width, scaled_height))

# Removed background scaling since we're using a white background
# scaled_background = pygame.transform.scale(background_image, (width, height))

# Scale the dragon image
dragon_scale_factor = 0.1
scaled_dragon = scale_image(dragon_image, dragon_scale_factor)
dragon_width, dragon_height = scaled_dragon.get_size()

# Scale egg images
egg_scale_factor = 0.3
scaled_eggs = [scale_image(img, egg_scale_factor) for img in egg_images]

# Ensure all egg images are the same size
obstacle_size = (50, 50)  # Standard size for obstacles
scaled_eggs = [pygame.transform.scale(img, obstacle_size) for img in scaled_eggs]

# Scale fireball image
fireball_scale_factor = 0.3
scaled_fireball = pygame.transform.scale(fireball_image, obstacle_size)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialize camera
cap = cv2.VideoCapture(0)

# Define obstacle types
OBSTACLE_EGG = 'egg'
OBSTACLE_FIREBALL = 'fireball'

# Define the number of columns
NUM_COLUMNS = 8  # Adjust as needed for your screen width

# Calculate the width of each column based on the screen width
COLUMN_WIDTH = width // NUM_COLUMNS

# Generate a list of possible x positions (center of each column)
possible_x_positions = [COLUMN_WIDTH * i + (COLUMN_WIDTH - obstacle_size[0]) // 2 for i in range(NUM_COLUMNS)]

# Function to create a new obstacle
def create_obstacle():
    # Get the list of x positions currently occupied by active obstacles
    occupied_x = [obstacle['rect'].x for obstacle in obstacles]

    # Determine available x positions by excluding occupied ones
    available_x = [x for x in possible_x_positions if x not in occupied_x]

    # If no positions are available, allow overlapping by resetting available_x
    if not available_x:
        available_x = possible_x_positions.copy()

    # Choose a random x position from the available ones
    x_pos = random.choice(available_x)

    # Randomly select the type of obstacle
    obstacle_type = random.choice([OBSTACLE_EGG, OBSTACLE_FIREBALL])

    # Initialize the obstacle dictionary
    obstacle = {'rect': pygame.Rect(x_pos, 0, obstacle_size[0], obstacle_size[1]),
                'type': obstacle_type}

    # If the obstacle is an egg, assign a random egg image
    if obstacle_type == OBSTACLE_EGG:
        obstacle['image'] = random.choice(scaled_eggs)

    return obstacle

# Obstacle settings
obstacles = []  # List to hold obstacles
obstacle_speed = 5  # Speed at which obstacles move down

# Initialize obstacles with reduced count
for _ in range(3):  # Start with 3 obstacles for better gameplay
    obstacles.append(create_obstacle())

# Define cooldown settings for obstacle spawning
obstacle_cooldown = 1500  # Time in milliseconds between spawns (1.5 seconds)
last_obstacle_spawn_time = pygame.time.get_ticks()  # Initialize spawn time

# Initialize game variables
running = True
game_over = False
score = 0
start_time = time.time()
game_duration = 120  # 2 minutes in seconds

# Initialize Clock for controlling frame rate
clock = pygame.time.Clock()

# Main loop
while running:
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = max(0, int(game_duration - elapsed_time))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and game_over:
            if event.key == pygame.K_r:  # Restart the game if 'R' is pressed
                score = 0
                obstacles.clear()
                for _ in range(3):  # Initialize with 3 obstacles upon restart
                    obstacles.append(create_obstacle())
                dragon_x = (width - dragon_width) // 2
                game_over = False
                start_time = time.time()
                last_obstacle_spawn_time = pygame.time.get_ticks()  # Reset spawn time

    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    # Default dragon position (centered)
    dragon_x = (width - dragon_width) // 2

    # Update dragon position based on hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
            # Map the wrist position to the window width
            dragon_x = int((wrist_x / frame.shape[1]) * width - dragon_width // 2)
            dragon_x = max(0, min(dragon_x, width - dragon_width))  # Keep within window bounds
            break  # Use the first detected hand

    if not game_over and remaining_time > 0:
        # Fill the screen with white color
        screen.fill((255, 255, 255))  # White background

        # Handle obstacle spawning based on cooldown
        current_ticks = pygame.time.get_ticks()
        if current_ticks - last_obstacle_spawn_time > obstacle_cooldown:
            obstacles.append(create_obstacle())
            last_obstacle_spawn_time = current_ticks

        # Update and draw obstacles
        for obstacle in obstacles[:]:
            obstacle['rect'].y += obstacle_speed  # Move the obstacle down

            # Select image based on obstacle type
            if obstacle['type'] == OBSTACLE_EGG:
                # Use the specific image assigned to this egg
                screen.blit(obstacle['image'], (obstacle['rect'].x, obstacle['rect'].y))
            else:
                screen.blit(scaled_fireball, (obstacle['rect'].x, obstacle['rect'].y))

            # Adjusted collision rectangle (excluding top 4% and sides 2%)
            collision_offset_top_percentage = 0.04  # 4%
            collision_offset_side_percentage = 0.02  # 2%

            collision_offset_top = int(dragon_height * collision_offset_top_percentage)
            collision_offset_side = int(dragon_width * collision_offset_side_percentage)

            dragon_rect = pygame.Rect(
                dragon_x + collision_offset_side,  # Shift right by 2% of width
                height - dragon_height - 10 + collision_offset_top,  # Shift down by 4% of height
                dragon_width - 2 * collision_offset_side,  # Reduce width by 4% (2% each side)
                dragon_height - collision_offset_top  # Reduce height by 4%
            )

            # Check for collision with the dragon
            if obstacle['rect'].colliderect(dragon_rect):
                if obstacle['type'] == OBSTACLE_EGG:
                    score += 1  # Increase score
                    obstacles.remove(obstacle)
                    obstacles.append(create_obstacle())
                elif obstacle['type'] == OBSTACLE_FIREBALL:
                    game_over = True  # End game

            # If the obstacle goes off the screen, reset its position
            if obstacle['rect'].y > height:
                obstacles.remove(obstacle)
                obstacles.append(create_obstacle())
                if obstacle['type'] == OBSTACLE_EGG:
                    score += 1  # Optionally increase score if egg is missed
                # Fireballs do not affect the score when missed

        # Draw the dragon image at the calculated position
        screen.blit(scaled_dragon, (dragon_x, height - dragon_height - 10))  # Slightly above the bottom

        # Draw the score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f'Score: {score}', True, (0, 0, 0))
        screen.blit(score_text, (10, 10))  # Top-left corner

        # Draw the timer
        timer_text = font.render(f'Time Left: {remaining_time}s', True, (0, 0, 0))
        screen.blit(timer_text, (width - 200, 10))  # Top-right corner

    else:
        if not game_over:
            # Time's up
            game_over = True
            end_text = "Time's Up!"
        else:
            end_text = "Game Over!"

        # Game Over screen
        screen.fill((0, 0, 0))  # Black background
        font_large = pygame.font.SysFont(None, 72)
        font_small = pygame.font.SysFont(None, 36)
        game_over_text = font_large.render(end_text, True, (255, 0, 0))
        final_score_text = font_small.render(f'Final Score: {score}', True, (255, 255, 255))
        restart_text = font_small.render('Press R to Restart', True, (255, 255, 255))
        screen.blit(game_over_text, (width // 2 - game_over_text.get_width() // 2, height // 2 - 100))
        screen.blit(final_score_text, (width // 2 - final_score_text.get_width() // 2, height // 2))
        screen.blit(restart_text, (width // 2 - restart_text.get_width() // 2, height // 2 + 50))

    # Update the display
    pygame.display.flip()

    # Control frame rate
    clock.tick(30)  # Limit to 30 frames per second

    # Show the camera feed with the drawn landmarks (optional)
    # Comment out the following lines if you don't want to see the camera feed
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw finger joints
    cv2.imshow('Camera Feed', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()
