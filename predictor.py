import subprocess
import cv2
import math
from cvzone.HandTrackingModule import HandDetector
import itertools
import numpy as np
import tensorflow as tf
from gtts import gTTS
from mutagen.mp3 import MP3
import threading
import pygame
import time

language = 'en'
width = 960
height = 540
log = False
flag = 0
counter = 0
offset = 20
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Hello', 'Hey', 'What\'s Up', 'My', 'Name is']
said = False
text = ""
previous_alphabet = ""
previous_word = ""
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)


model = tf.keras.models.load_model('gestureswithalphabets.h5')

def normalize_landmarks(landmarks, center):
    """
    Normalizes the landmarks relative to the center of the hand.
    """
    normalized_landmarks = []
    original_landmarks = []
    for lm in landmarks:
        x_org = lm[0]
        y_org = lm[1]
        x_norm = (lm[0] - center[0]) / math.sqrt((center[0]**2 + center[1]**2))
        y_norm = (lm[1] - center[1]) / math.sqrt((center[0]**2 + center[1]**2))
        normalized_landmarks.append([x_norm, y_norm])
        original_landmarks.append([x_org, y_org])
    return normalized_landmarks, original_landmarks

def getPrediction(landmarks, model):
    """
    Returns the prediction and confidence of the model.
    """
    prediction = model.predict(np.array([landmarks]), verbose=0)
    confidence = np.amax(prediction)
    index = np.argmax(prediction)
    return index, confidence

def speak(text):
    """
    Converts the text to speech.
    """
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("word.mp3")
    audio = MP3("word.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("word.mp3")
    pygame.mixer.music.play()
    time.sleep(audio.info.length)
    pygame.quit()
    subprocess.Popen(['mpg123', 'word.mp3'])

while True: 
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, _ = detector.findHands(img, flipType=False)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        hand_landmarks = hand['lmList']
        hand_center = hand['center']
        normalized_landmarks, original_landmarks = normalize_landmarks(hand_landmarks, hand_center)
    
        conv_lst = list(itertools.chain.from_iterable(normalized_landmarks))
        new_lm = np.concatenate((conv_lst, np.zeros(42)))
    
        if len(hands) > 1:
            hand2 = hands[1]
            hand2_landmarks = hand2['lmList']
            hand2_center = hand2['center']
            normalized_landmarks2, original_landmarks2 = normalize_landmarks(hand2_landmarks, hand2_center)

            conv_lst2 = list(itertools.chain.from_iterable(normalized_landmarks2))
            new_lm = np.concatenate((conv_lst2, conv_lst))
            
        index, confidence = getPrediction(new_lm, model)
        predictedText = labels[index]
        
        if confidence > 0.90 and len(predictedText) == 1:
            if predictedText and predictedText != previous_alphabet:
                text += predictedText
                previous_alphabet = predictedText

        elif len(predictedText) > 1:
            word = predictedText
            word_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
            window_width = img.shape[1]
            word_x = int((window_width - word_size[0]) / 2)
            word_y = int(word_size[1])
            cv2.putText(img, word, (word_x, word_y + 50 ), cv2.FONT_HERSHEY_COMPLEX, 1.7, (7, 21, 219), 2)
            
        if len(predictedText) == 1:
            cv2.putText(img, labels[index], (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1.7, (7, 21, 219), 2)
            cv2.putText(img, "{:.2f}%".format(confidence*100), (70, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (7, 21, 219), 2)
        
    if log == True:
        cv2.putText(img, text, (300, 135), cv2.FONT_HERSHEY_COMPLEX, 1.7, (7, 21, 219), 2)    
    cv2.imshow("Image", img) 

    key = cv2.waitKey(10)
    if key == ord("q") or key == 27:
        break

    elif key == ord("D"):
        text = ""

    elif key == ord("d"):
        text = text[:-1]

    elif key == ord("l"):
        log = True
    
    elif key == ord("L"):
        log = False

    elif key == ord("s"):
        if text and said == False:
            speak(text)
            said = True
    said = False

cap.release()
cv2.destroyAllWindows()

