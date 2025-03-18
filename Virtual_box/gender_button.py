import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
from time import sleep
import numpy as np

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
                        elif button.text == "backspace":
                            age = age[:-1]  # Delete last character
                        else:
                            # Append selected age digit only if the resulting age is <= 100
                             if age == "" or int(age + button.text) <= 100:
                                 age += button.text  # Append selected age digit

    # Display Selected Text
    cv2.putText(img, f"Gender: {gender}", (20, 650), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 4)
    cv2.putText(img, f"Age: {age}", (20, 700), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 4)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
