import cv2
import os
import mediapipe as mp
import torch
from ultralytics import YOLO  
from database import add_data_to_database  # Import the function


# Load YOLOv8 model
model =  YOLO("Testing_Result/yolov8m/train13/weights/best.pt")

# Initialize Mediapipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height


# Load the background image
imgBackground = cv2.imread("GUI/Resources/Background2.png")

# Load mode images
folderPathModes = "GUI/Resources/Modes"
listImgModesPath = os.listdir(folderPathModes)
listImgModes = [cv2.imread(os.path.join(folderPathModes, imgModePath)) for imgModePath in listImgModesPath]

modeType = 0   # for changing selection mode
selection = -1
counter = 0
selectionSpeed = 15
gesture_position = [(1184,222),(946,341),(1186,454),(933,565),(1184,648)]
counterPause = 0

answer = ["very satisfied", "satisfied", "normal","dissatisfied","very dissatisfied"]  # the answer will store in a list then store in database
database = []


# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Overlay the processed webcam feed on the background image
        imgBackground[139:139+480, 50:50+640] = frame
        imgBackground[0:720, 847:1280] = listImgModes[modeType]

        if not ret:
            break

        # Process frame with YOLOv8
        yolo_results = model(frame)

        # Print YOLOv8 results
        most_confident_box = None

        for result in yolo_results:
            boxes = result.boxes
            if boxes:
                # Find the most confident detection
                most_confident_box = max(boxes, key=lambda box: box.conf[0])

        if most_confident_box and counterPause==0 and modeType < 6:
            x1, y1, x2, y2 = map(int, most_confident_box.xyxy[0])  # Extract bounding box coordinates
            confidence = most_confident_box.conf[0]  # Extract confidence score
            class_id = most_confident_box.cls[0]  # Extract class id

            # Print the most confident detection's details
            print(f'Class: {model.names[int(class_id)]}, Confidence: {confidence:.2f}, Box: ({x1}, {y1}, {x2}, {y2})')

            # Draw bounding box and label on the frame
            label = f'{model.names[int(class_id)]}: {confidence:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if model.names[int(class_id)] == "verysatisfied": # "verysatisfied" is depend on the class name during annotated at roboflow
                if selection != 1:
                    counter = 1
                selection = 1
            elif model.names[int(class_id)] == "satisfied":
                if selection != 2:
                    counter = 1
                selection = 2
            elif model.names[int(class_id)] == "normal":
                if selection != 3:
                    counter = 1
                selection = 3
            elif model.names[int(class_id)] == "dissatified":
                if selection != 4:
                    counter = 1
                selection = 4
            elif model.names[int(class_id)] == "very dissatisfied":
                if selection != 5:
                    counter = 1
                selection = 5
            else:
                selection= -1
                counter = 0
            if counter>0:
                counter +=1

                cv2.ellipse(imgBackground,gesture_position[selection-1],(67,67),0,0, 
                            counter*selectionSpeed,(0,255,0), 10)

                if counter*selectionSpeed>360:
                    database.append(answer[selection-1])
                    modeType +=1
                    counter=0
                    selection=-1
                    counterPause=1
                    print(database)

        if counterPause>0:    # to make break between the questions
            counterPause+=1
            if counterPause>60:    # 60 = 2second because 30 per frame
                counterPause=0

        if modeType == 5:
            add_data_to_database(database)
            print(database)

            break

        # Recolor feed for holistic model processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Draw holistic model results on the frame
        # Right hand
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # Left hand
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Display the background with the overlay
        cv2.imshow("Background", imgBackground)
        # Display Selected Text
        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        


    cv2.destroyAllWindows()