from types import NoneType
import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
from tensorflow import keras
from keras.models import load_model

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Color Conversion BGR to RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make Prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Color Conversion RGB to BGR
    return image, results

def draw_styled_landmarks(image, results):
    
    # Draw Face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), # dot color
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))# line color
    
    # Draw Pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(99,153,20), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(18,100,181), thickness=2, circle_radius=4))       
    
    # Draw Left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=4))
    
    # Draw Right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(168,97,25), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(180,56,21), thickness=2, circle_radius=4)) 


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132) # 33*4
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63) # 21*3
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63) # 21*3
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404) # 468*3 
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

mp_holistic = mp.solutions.holistic  # Holistic Model for detections
mp_drawing = mp.solutions.drawing_utils # Drawing utilities for drawing detections
actions = np.array(['hello','thanks','iloveyou'])
colors = [(245,117,16), (117,245,16), (16,117,245)]
model = load_model("actions.h5")   # my model : actions.h5

st.title('Sign Language Translator')

run = st.checkbox('run')
frame_window = st.image([])
cap = cv2.VideoCapture(0)
while run:
    
    sequence = []
    sentence = []
    threshold = 0.7
    res = np.array([1,0,0])
    # Access Mediapipe Model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():

            # read feed 
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            
            # Make Detections
            image, results = mediapipe_detection(frame, holistic)
                
            # Draw landmarks
            #draw_styled_landmarks(image, results)

            # Predection Logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #print(actions[np.argmax(res)])
            
            # Vizualizing Logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if(actions[np.argmax(res)] != sentence[-1]):
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
            
            if len(sentence) > 5:
                sentence = sentence[-5:]
                        
            # Viz Probabilities
            image = prob_viz(res, actions, image, colors)
                        
            cv2.rectangle(image, (0,0), (640,40), (245, 117, 69), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
            # show frame to screen
            #cv2.imshow('OpenCV feed',image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_window.image(image)

            # quit capturing
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

                # release camera        
                #cap.release()

                # close all the windows
                #cv2.destroyAllWindows()
else:
    cap.release()



