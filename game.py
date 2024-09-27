import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp
import random

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))  # Windowed mode
width, height = screen.get_size()  # Get current window size

# Load the background image from the assets folder
background_image = pygame.image.load('assets/background.jpg')  # Your background image
21
# Scale the background to fit the window
background_image = pygame.transform.scale(background_image, (width, height))

# Load the dragon image from the assets folder
dragon_image = pygame.image.load('assets/dragon-removebg.png')  # Your dragon image

# Load the obstacle image from the assets folder
obstacle_image = pygame.image.load('assets/bird.webp')  # Your obstacle image
obstacle_image = pygame.transform.scale(obstacle_image, (60, 60))  # Scale to desired size

# Scale the dragon image
img_width, img_height = dragon_image.get_size()
scale_factor = 0.3  # Adjust as necessary
scaled_width = int(width * scale_factor)
scaled_height = int((img_height / img_width) * scaled_width)
scaled_dragon_image = pygame.transform.scale(dragon_image, (scaled_width, scaled_height))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialize camera
cap = cv2.VideoCapture(0)

# Function to create a new obstacle
def create_obstacle():
    x_pos = random.choice([0, width // 3, 2 * width // 3])
    y_pos = 0  # Start at the top
    return pygame.Rect(x_pos, y_pos, 50, 50)  # Rect for collision detection

# Obstacle settings
obstacles = []  # List to hold obstacles
obstacle_speed = 5  # Speed at which obstacles move down
for _ in range(3):  # Create 3 obstacles initially
    obstacles.append(create_obstacle())

# Initialize game variables
running = True
dragon_x = (width - scaled_width) // 2  # Center it at the beginning
score = 0
game_over = False

# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and game_over:
            if event.key == pygame.K_r:  # Restart the game if 'R' is pressed
                score = 0
                obstacles.clear()
                for _ in range(3):
                    obstacles.append(create_obstacle())
                dragon_x = (width - scaled_width) // 2
                game_over = False

    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])

            # Map the wrist position to the window width
            dragon_x = int((wrist_x / frame.shape[1]) * width - scaled_width // 2)
            dragon_x = max(0, min(dragon_x, width - scaled_width))  # Keep within window bounds

    if not game_over:
        # Draw the background image
        screen.blit(background_image, (0, 0))  # Blit the background image at (0, 0)

        # Update and draw obstacles
        for obstacle in obstacles:
            obstacle.y += obstacle_speed  # Move the obstacle down
            screen.blit(obstacle_image, (obstacle.x, obstacle.y))  # Draw the obstacle image

            # Check for collision with the dragon
            if obstacle.colliderect(pygame.Rect(dragon_x, height - scaled_height, scaled_width, scaled_height)):
                game_over = True  # Set game over when collision occurs

            # If the obstacle goes off the screen, reset its position
            if obstacle.y > height:
                obstacles.remove(obstacle)
                obstacles.append(create_obstacle())  # Create a new obstacle
                score += 1  # Increment score for each obstacle successfully avoided

        # Draw the scaled dragon image at the calculated position
        screen.blit(scaled_dragon_image, (dragon_x, height - scaled_height))

        # Draw the score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f'Score: {score}', True, (0, 0, 0))
        screen.blit(score_text, (10, 10))  # Position of the score in the top left corner

    else:
        # Game Over screen
        font = pygame.font.SysFont(None, 72)
        game_over_text = font.render('Game Over!', True, (255, 0, 0))
        restart_text = font.render('Press R to Restart', True, (0, 0, 0))
        screen.blit(game_over_text, (width // 2 - 150, height // 2 - 50))
        screen.blit(restart_text, (width // 2 - 170, height // 2))

    # Update the display
    pygame.display.flip()

    # Show the camera feed with the drawn landmarks (optional)
    cv2.imshow('Camera Feed', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()
