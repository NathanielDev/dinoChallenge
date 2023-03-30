import tensorflow as tf
import numpy as np
import pyautogui
import cv2

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with appropriate loss and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Collect training data
X_train = []
y_train = []
for i in range(1000):
    # Capture screen and extract features
    screenshot = pyautogui.screenshot(region=(0, 0, 600, 150))
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    obstacle = screenshot[50:75, 450:480]
    obstacle_mean = obstacle.mean()
    speed = pyautogui.locateOnScreen('speed.png', region=(0, 100, 600, 50))
    if speed:
        speed_x = speed.left + 20
        speed_y = speed.top + 5
        speed_color = screenshot[speed_y, speed_x]
        speed_feature = speed_color / 255.0
    else:
        speed_feature = 0
    
    # Determine whether to jump or not based on distance and speed
    if obstacle_mean < 200:
        y = 1
    elif obstacle_mean < 225 and speed_feature > 0.5:
        y = 1
    else:
        y = 0
    
    # Add features and label to training data
    X_train.append([obstacle_mean, speed_feature])
    y_train.append(y)

# Preprocess the training data
X_train = np.array(X_train)
y_train = np.array(y_train)

# Train the neural network
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Test the neural network on the game
while True:
    screenshot = pyautogui.screenshot(region=(0, 0, 600, 150))
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    obstacle = screenshot[50:75, 450:480]
    obstacle_mean = obstacle.mean()
    speed = pyautogui.locateOnScreen('speed.png', region=(0, 100, 600, 50))
    if speed:
        speed_x = speed.left + 20
        speed_y = speed.top + 5
        speed_color = screenshot[speed_y, speed_x]
        speed_feature = speed_color / 255.0
    else:
        speed_feature = 0
    
    # Make prediction using neural network
    X_test = np.array([[obstacle_mean, speed_feature]])
    y_pred = model.predict(X_test)[0][0]
    
    # Jump if predicted probability is greater than 0.5
    if y_pred > 0.5:
        pyautogui.press('up')
