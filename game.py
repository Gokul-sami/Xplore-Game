import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))  # Windowed mode
width, height = screen.get_size()  # Get current window size

# Load the background image from the assets folder
background_image = pygame.image.load('assets/dragon-removebg.png')  # Your dragon image

# Scale the dragon image
img_width, img_height = background_image.get_size()
scale_factor = 0.5  # Adjust as necessary
scaled_width = int(width * scale_factor)
scaled_height = int((img_height / img_width) * scaled_width)
scaled_image = pygame.transform.scale(background_image, (scaled_width, scaled_height))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize dragon_x with a default value
dragon_x = (width - scaled_width) // 2  # Center it at the beginning

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

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
            # Draw landmarks
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw finger joints

            # Get the wrist position for tracking
            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
            wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

            # Map the wrist position to the window width
            dragon_x = int((wrist_x / frame.shape[1]) * width - scaled_width // 2)
            dragon_x = max(0, min(dragon_x, width - scaled_width))  # Keep within window bounds

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the scaled dragon image at the calculated position
    screen.blit(scaled_image, (dragon_x, height - scaled_height))

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
