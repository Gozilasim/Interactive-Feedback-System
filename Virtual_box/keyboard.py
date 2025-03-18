import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
from time import sleep
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#set the confidence level with 80%
detector = HandDetector(detectionCon=0.8, maxHands=2)
keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"]
        ]

rightButton = cv2.imread("Virtual_box/Button_image/right-arrow.png", cv2.IMREAD_UNCHANGED)
lefttButton = cv2.flip(rightButton,1)



def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),
                       (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 40, y + 60),
                     cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

def nextButton(img, buttonList):
    
    for button in buttonList:
        x,y = button.pos
        w,h = button.size
        cv2.rectangle(img, button.pos,(x+w, y+h),(160,160,160), cv2.FILLED)
        cv2.putText(img, button.text,(x+ 25,y+65), cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)

    return img

class Button():
    def __init__(self, pos, text, size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text




buttonList = []
finalText = []

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100*j+50, 100*i +50],key))

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    lmList = []
    if hands:
        #hand 1
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

            print(lmList1)

           

    

    img = drawAll(img, buttonList)

    if lmList:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x<lmList[8][0] <x+w and y<lmList[8][1] <y+h:
                    cv2.rectangle(img, button.pos,(x+w, y+h),(175,0,175), cv2.FILLED)
                    cv2.putText(img, button.text,(x+ 25,y+65), cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
                
                    length, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
                    print(length)

                    # when clicked
                    if length < 30:
                        cv2.rectangle(img, button.pos,(x+w, y+h),(0,250,0), cv2.FILLED)
                        cv2.putText(img, button.text,(x+ 25,y+65), cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)

                        finalText.append(button.text)
                        sleep(0.15)

                        if button.text == "/":
                            finalText.clear()


       
    cv2.rectangle(img, (50, 350),(700,450),(175,0,175), cv2.FILLED)
    cv2.putText(img, str(finalText),(60,425), cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)

    # Define the new size (for example, width=100 and height=50)
    new_width = 100
    new_height = 100
    new_size = (new_width, new_height)

    # Resize the right arrow image
    rightButton_resized = cv2.resize(rightButton, new_size, interpolation=cv2.INTER_AREA)

    # Resize the left arrow image
    leftButton_resized = cv2.resize(lefttButton, new_size, interpolation=cv2.INTER_AREA)


    img = cvzone.overlayPNG(img,rightButton_resized,(1150,350))
    img = cvzone.overlayPNG(img,leftButton_resized,(72,350))


    cv2.imshow("Image", img)
    cv2.waitKey(1)
