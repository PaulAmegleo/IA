import cv2
import mediapipe as mp
import csv
import os
from joblib import load
import pandas as pd
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
bbox_x_min = 50
bbox_y_min = 100
bbox_x_max = 400
bbox_y_max = 400

cap = cv2.VideoCapture(0)
capture_flag = False




with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            capture_flag = True
            training_data = []
            with open('Data.csv', 'a', newline='') as files:
                writer = csv.writer(files)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(21):
                            point = hand_landmarks.landmark[i]
                            x = point.x
                            y = point.y
                            z = point.z

                            training_data.extend([x, y, z])       
                    writer.writerow(training_data)  # Writing X, Y, Z coordinates and gesture to CSV
            
        else:
            capture_flag = False

        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (bbox_x_min, bbox_y_min), (bbox_x_max, bbox_y_max), (0, 255, 0), 2)  



        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z  
                if bbox_x_min < x * image.shape[1] < bbox_x_max and bbox_y_min < y * image.shape[0] < bbox_y_max:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    data = {}
                    for i in range(21):
                        point = hand_landmarks.landmark[i]
                        data[f"x{i}"] = point.x
                        data[f"y{i}"] = point.y
                        data[f"z{i}"] = point.z
                    df = pd.DataFrame(data, index=[0])
                    loaded_model = load("modelBinary.joblib")
                    loadedPrediction = loaded_model.predict(df)
                    print(loadedPrediction[0])  


                    if loadedPrediction[0] == "Up":
                     pyautogui.press("up")
                    elif loadedPrediction[0] == "Down":
                     pyautogui.press("down")
                    elif loadedPrediction[0] == "Left":
                     pyautogui.press("left")
                    elif loadedPrediction[0] == "Right":
                     pyautogui.press("right")
                  
    


        cv2.imshow('MediaPipe Hands', image,)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
