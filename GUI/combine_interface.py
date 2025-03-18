import cv2
import os
import mediapipe as mp
import torch
from ultralytics import YOLO  
from database import add_data_to_database  
from cvzone.HandTrackingModule import HandDetector
import cvzone
from time import sleep
import numpy as np

def run_feedback_system():

    # Load YOLOv8 model
    model = YOLO("Testing_Result/yolov8m/train13/weights/best.pt")

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

    answer = ["very satisfied", "satisfied", "normal","dissatified","very dissatisfied"] # the answer will store in a list then store in database
    database = []
    database.append(gender)  
    database.append(age)

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

                if model.names[int(class_id)] == "verysatisfied":  # "verysatisfied" is depend on the class name during annotated at roboflow
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

            if modeType == 5:      # the last question
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
            cv2.putText(img, f"Gender: {gender}", (62, 535), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
            cv2.putText(img, f"Age: {age}", (62, 585), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        


    cv2.destroyAllWindows()



# Load the images
male_img = cv2.imread('E:/UTeM2/FYP1/project/Gesture_recognition/pic/male-Photoroom-transformed.png')
female_img = cv2.imread('E:/UTeM2/FYP1/project/Gesture_recognition/pic/female-Photoroom-transformed.png')

# Keypad layout
keys = [["7", "8", "9"],
        ["4", "5", "6"],
        ["1", "2", "3"],
        ["0", "backspace", "Confirm"]]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Set the confidence level to 80%
detector = HandDetector(detectionCon=0.8, maxHands=2)

def draw_gender(img, buttonList, lmList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size

        # Check if the finger is over the button
        if lmList and x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
            # Enlarge image by 1.2x
            scale = 1.2
            enlarged_img = None
            if button.text == "Male":
                enlarged_img = cv2.resize(male_img, (int(w * scale), int(h * scale)))
            elif button.text == "Female":
                enlarged_img = cv2.resize(female_img, (int(w * scale), int(h * scale)))

            if enlarged_img is not None:
                # Calculate the new position to keep the image centered
                new_x = x - int((w * scale - w) / 2)
                new_y = y - int((h * scale - h) / 2)
                img[new_y:new_y+enlarged_img.shape[0], new_x:new_x+enlarged_img.shape[1]] = enlarged_img
        else:
            # Display original image with transparency
            if button.text == "Male":
                imgNew[y:y+male_img.shape[0], x:x+male_img.shape[1]] = male_img
            elif button.text == "Female":
                imgNew[y:y+female_img.shape[0], x:x+female_img.shape[1]] = female_img
            else:
                # Draw other types of buttons (if any)
                cvzone.cornerRect(imgNew, (x, y, w, h), 20, rt=0)
                cv2.rectangle(imgNew, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgNew, button.text, (x + 40, y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    # Apply transparency only to the overlay
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

def draw_age(img, buttonList, lmList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size

        # Check if the finger is over the button
        if lmList and x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
            # Enlarge button by 1.2x
            scale = 1.2
            enlarged_size = (int(w * scale), int(h * scale))
            enlarged_x = x - int((enlarged_size[0] - w) / 2)
            enlarged_y = y - int((enlarged_size[1] - h) / 2)

            cvzone.cornerRect(imgNew, (enlarged_x, enlarged_y, enlarged_size[0], enlarged_size[1]), 20, rt=0)
            cv2.rectangle(img, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
            cv2.putText(img, button.text, (x + 25, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
        else:
            # Draw the button normally
            cvzone.cornerRect(imgNew, (x, y, w, h), 20, rt=0)
            cv2.rectangle(imgNew, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgNew, button.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    # Apply transparency only to the overlay
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

class gender_Button():
    def __init__(self, pos, text, size=[314, 314]):
        self.pos = pos
        self.size = size
        self.text = text

class age_Button():
    def __init__(self, pos, text, size=[130, 130]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []
gender_selected = False
gender = ""
age = ""

# Initialize Gender buttons
buttonList.append(gender_Button([196, 204], "Male"))
buttonList.append(gender_Button([790, 204], "Female"))

def create_age_buttons():
    global buttonList
    buttonList = []
    initial_x = 400  # Starting x position
    initial_y = 100  # Starting y position
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(age_Button([initial_x + 200*j, initial_y + 150*i], key))



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    lmList = []
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        lmList = lmList1

    # Gender Selection Stage
    if not gender_selected:
        img = draw_gender(img, buttonList, lmList)
    else:
        # Age Selection Stage
        img = draw_age(img, buttonList, lmList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                length, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)

                # When clicked
                if length < 30:
                    sleep(0.3)
                    if not gender_selected:
                        gender = button.text
                        gender_selected = True
                        create_age_buttons()
                    else:
                        if button.text == "Confirm":
                            print(f"Gender: {gender}, Age: {age}")
                            cv2.destroyAllWindows()
                            run_feedback_system()
                        elif button.text == "backspace":
                            age = age[:-1]  # Delete last character
                        else:
                            # Append selected age digit only if the resulting age is <= 100
                             if age == "" or int(age + button.text) <= 100:
                                 age += button.text  # Append selected age digit

    # Display Selected Text
    cv2.putText(img, f"Gender: {gender}", (20, 650), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
    cv2.putText(img, f"Age: {age}", (20, 700), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)

    cv2.imshow("Image", img)
    cv2.waitKey(1)



 
