import cv2
import math
from cvzone.HandTrackingModule import HandDetector
import itertools
import csv
import numpy as np
counter = 0
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M': 12, 'N':13,
          'O':14, 'P':15, 'Q':16, 'R':17, 'S':18, 'T':19, 'U':20, 'V': 21, 'W':22, 'X':23, 'Y':24, 'Z':25, 'Hello':26, 
          'Hey':27, "What's Up": 28, 'My':29, 'Name is':30,}

def normalize_landmarks(landmarks, center):
    """
    Normalizes the landmarks relative to the center of the hand.
    """
    normalized_landmarks = []
    for lm in landmarks:
        x_norm = (lm[0] - center[0]) / math.sqrt((center[0]**2 + center[1]**2))
        y_norm = (lm[1] - center[1]) / math.sqrt((center[0]**2 + center[1]**2))
        normalized_landmarks.append([x_norm, y_norm])
    return normalized_landmarks

alpha = input("Enter the alphabet to collect data.\n")
id = labels[alpha]

while True: 
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)
    
    if hands:
        hand = hands[0]
        hand_landmarks = hand['lmList']
        hand_center = hand['center']
        normalized_landmarks = normalize_landmarks(hand_landmarks, hand_center)
        conv_lst = list(itertools.chain.from_iterable(normalized_landmarks))

        if len(hands) >1:
            hand2 = hands[1]
            hand2_landmarks = hand2['lmList']
            hand2_center = hand2['center']
            normalized_landmarks2 = normalize_landmarks(hand2_landmarks, hand2_center)
            conv_lst1 = list(itertools.chain.from_iterable(normalized_landmarks2))
            new_lm = np.concatenate((conv_lst, conv_lst1))
            print(new_lm,  len(new_lm))
            
            with open('hand_landmarks.csv', 'a', newline="") as f:
                counter+=1
                writer = csv.writer(f)
                writer.writerow([id, *new_lm])
            print(counter)
               

        else:
            conv_lst = list(itertools.chain.from_iterable(normalized_landmarks))
            zeros = np.zeros(42)
            new_lm = np.concatenate((conv_lst, zeros))
            

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
            counter+=1
            with open('hand_landmarks.csv', 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([id, *new_lm])
            print(counter)
            
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()