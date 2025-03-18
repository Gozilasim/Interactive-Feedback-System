import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
from time import sleep
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Set the confidence level to 80%
detector = HandDetector(detectionCon=0.8, maxHands=2)

# E:/UTeM2/FYP1/project/Gesture_recognition/pic/male-Photoroom-transformed.png
#  E:/UTeM2/FYP1/project/Gesture_recognition/pic/female-Photoroom-transformed.png
keys = [["7", "8", "9", ],
         ["4", "5", "6", ],
         ["1", "2", "3", ],
         ["0", "⌫" ,"Confirm"]]

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 40, y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []
finalText = []
isUppercase = True  # Flag to track if uppercase is active



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    lmList = []
    if hands:
        # hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1["center"]
        handType1 = hand1["type"]

        lmList = lmList1
        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            centerPoint2 = hand2["center"]
            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)

    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                cv2.rectangle(img, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 25, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                length, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
                print(length)

                # when clicked
                if length < 30:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 250, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 25, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

 
                    finalText.append(button.text)
                    sleep(0.3)

    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, str(finalText), (60, 425), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
