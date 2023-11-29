from flask import Flask, render_template, Response,url_for,redirect,request, session, send_file
import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient
import gridfs
###
from flask import Flask, request, render_template, redirect, url_for
from flask_mysqldb import MySQL
import os
from werkzeug.utils import secure_filename
import io
import mysql.connector
###
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
# from mediapipe import Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
# from mediapipe.solutions import Holistic

from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from englisttohindi.englisttohindi import EngtoHindi
import pygame

app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera2 = ""
data_index = ""

fetch_name=""
type = "face"

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Umang123@'
app.config['MYSQL_DB'] = 'mydatabase'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

##########################################################################################

# #get_files
# out_data = ""
# def upload_file(file_loc, file_name, fs):
#     with open(file_loc, 'rb') as file_data:
#         data = file_data.read()
#     fs.put(data, filename=file_name)

# def get_file(db, fs, file_name):
#     data = db.images.files.find_one({"filename": file_name})
#     fs_id = data['_id'] 
#     global out_data
#     out_data = fs.get(fs_id).read()

# file_loc = ".\photos\elon.jpg"
# file_name="elon.jpg"

# #connecting mongodb
# client = MongoClient()
# db = client.faces
# fs = gridfs.GridFS(db, collection="images")

# get_file(db, fs, file_name)

###############################################################################
#making encodings

##***imp*** --> dont delete
# import os

# def get_image_file_names_with_paths_and_names(folder_path):
#     file_info_list = []

#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith(".jpg"):
#                 relative_path = os.path.relpath(root, folder_path)
#                 if relative_path == ".":
#                     file_path = "photos/" + file
#                 else:
#                     file_path = "photos/" + os.path.join(relative_path, file)
#                 file_path = file_path.replace("\\", "/")
                
#                 filename_without_extension = os.path.splitext(file)[0]
                
#                 file_info_list.append({"path": file_path, "name": filename_without_extension})

#     return file_info_list

# folder_to_traverse = r"D:\PROJECTS\Python Projects\face-rec\photos"  # Replace with the actual folder path
# image_file_info_list = get_image_file_names_with_paths_and_names(folder_to_traverse)

# known_face_encodings = []
# known_face_names = []
# i = 0

# for file_info in image_file_info_list:
#     person_image = face_recognition.load_image_file(file_info["path"])
#     known_face_encodings.append(face_recognition.face_encodings(person_image)[0])
#     known_face_names.append(file_info["name"])
#     i = i + 1


##############################################################################
### Retreiving data from database ###

# # List to store face encodings and names
# known_face_encodings = []
# known_face_names = ["shaurya"]

# Lists to store face encodings, usernames, user_ids, and user_data
known_face_encodings = []
known_face_names = []  # List for usernames
known_user_ids = []  # List for user_ids
known_user_data = []  # List for user_data

# # Query to retrieve image data for all users
# with app.app_context():
#     cursor = mysql.connection.cursor()
#     cursor.execute("SELECT image_data FROM users")
#     image_data_list = cursor.fetchall()
#     cursor.close()


# # Process each image data
# for image_data in image_data_list:
#     person_image = face_recognition.load_image_file(io.BytesIO(image_data['image_data']))  # Convert bytes to image
#     person_face_encoding = face_recognition.face_encodings(person_image)[0]  # Assuming only one face per image

#     known_face_encodings.append(person_face_encoding)

# mysql_host = 'localhost'
# mysql_user = 'Shaurya'
# mysql_password = 'Umang123@'
# mysql_db = 'mydatabase'

# try:
#     cur = mysql.connection.cursor()
#     cur.execute("SELECT image_data FROM users WHERE image_data IS NOT NULL")
#     rows = cur.fetchall()
#     cur.close()

#     image_list = []
#     for row in rows:
#         image_data = row['image_data']
#         image = convert_image_data(image_data)
#         if image is not None:
#             image_list.append(image)

#     # Now, 'image_list' contains the images from the database

# except Exception as e:
#     print(f"Error: {str(e)}")

###################################################################################

# # Load a sample picture and learn how to recognize it.
# elon_image = face_recognition.load_image_file("photos/elon.jpg")
# elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

# # Load a second sample picture and learn how to recognize it.
# bill_image = face_recognition.load_image_file("photos/bill.jpg")
# bill_face_encoding = face_recognition.face_encodings(bill_image)[0]

# shaurya_image = face_recognition.load_image_file("photos/shaurya.jpg")
# shaurya_face_encoding = face_recognition.face_encodings(shaurya_image)[0]

# # Create arrays of known face encodings and their names
# known_face_encodings = [
#     elon_face_encoding,
#     bill_face_encoding,
#     shaurya_face_encoding
    
# ]
# known_face_names = [
#     "Elon",
#     "Bill",
#     "shaurya"
# ]

###################################################################################

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


def generate_frames():
    while True:
        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
                # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    global data_index
                    data_index = best_match_index
                #here we can make use of index to retrieve all other data

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                global fetch_name
                fetch_name=name
                #put all other related data in other global variables
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

no_sequences = 30
sequence_length = 30
            
## IMPORTANT DONT DELETE
# def sign():
#     global camera2
#     camera2 = cv2.VideoCapture(0)
#     while True:
#         success, frame2 = camera2.read()
#         if not success:
#             break
#         else:
#             small_frame = cv2.resize(frame2, (0, 0), fx=0.25, fy=0.25)
#             ret, buffer = cv2.imencode('.jpg', frame2)
#             frame2 = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

##############################################################################################################################
#### Please check from here
# Error: OpenCV(4.8.1) d:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.simd_helpers.hpp:94: error: (-2:Unspecified error) in function '__cdecl cv::impl::`anonymous-namespace'::CvtHelper<struct cv::impl::`anonymous namespace'::Set<3,4,-1>,struct cv::impl::A0x11a46be7::Set<3,4,-1>,struct cv::impl::A0x11a46be7::Set<0,2,5>,3>::CvtHelper(const class cv::_InputArray &,const class cv::_OutputArray &,int)'
# > Unsupported depth of input image:
# >     'VDepth::contains(depth)'
# > where
# >     'depth' is 6 (CV_64F)
#this is the error i am getting


# # # Load your pre-trained model
# model = keras.models.load_model('action.h5')  # Replace with your model path

# # # Define actions (replace with your actual action names)
# actions = ["action1", "action2", "action3"]

#Function for processing video frames and making predictions
# def sign():
#     global camera2
#     camera2 = cv2.VideoCapture(0)

#     with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             try:
#                 success, frame2 = camera2.read()

#                 if not success:
#                     break

#                 # Preprocess the frame (resize, convert to RGB, etc.) if required
#                 processed_frame = preprocess_frame(frame2)

#                 # Make predictions using your model
#                 recognized_action = predict_action(holistic, processed_frame)

#                 # Display the recognized action on the frame
#                 cv2.putText(frame2, recognized_action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#                 # Encode the frame as JPEG
#                 ret, buffer = cv2.imencode('.jpg', frame2)
#                 frame_data = buffer.tobytes()

#                 # Yield the frame data as a response
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
#             except Exception as e:
#                 print(f"Error: {str(e)}")


# def preprocess_frame(frame):
#     # Implement your frame preprocessing here (e.g., resizing, converting to RGB)
#     processed_frame = cv2.resize(frame, (224, 224))
#     processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
#     processed_frame = processed_frame / 255.0  # Normalize the frame

#     return processed_frame

# def preprocess_frame(frame):
#     # Check if the frame has 64-bit floating-point depth
#     if frame.dtype == np.float64:
#         # Convert the frame to 8-bit unsigned integer format
#         frame = cv2.convertScaleAbs(frame)

#     # Check if the frame is in grayscale
#     if len(frame.shape) == 2:
#         # Convert grayscale to BGR
#         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#     # Implement your frame preprocessing here (e.g., resizing, converting to RGB)
#     processed_frame = cv2.resize(frame, (224, 224))
#     processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
#     processed_frame = processed_frame / 255.0  # Normalize the frame

#     return processed_frame




# def predict_action(holistic, frame):
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = holistic.process(image)

#     if results.pose_landmarks:
#         keypoints = extract_keypoints(results)
#         keypoints = np.expand_dims(keypoints, axis=0)  # Add batch dimension

#         # Make predictions using your model
#         res = model.predict(keypoints)[0]
#         recognized_action = actions[np.argmax(res)]
#     else:
#         recognized_action = "No pose detected"

#     return recognized_action

# def sign():
#     global camera2
#     camera2 = cv2.VideoCapture(0)
#     while True:
#         success, frame2 = camera2.read()
#         if not success:
#             break
#         else:
#             small_frame = cv2.resize(frame2, (0, 0), fx=0.25, fy=0.25)
#             ret, buffer = cv2.imencode('.jpg', frame2)
#             frame2 = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


# holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# def sign():
#     global camera2
#     camera2 = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             success, frame2 = camera2.read()
            
#             if not success:
#                 break
#             else:
                
#                 # Resize the frame for better performance
#                 small_frame = cv2.resize(frame2, (0, 0), fx=0.25, fy=0.25)
                
#                 # Convert the BGR frame to RGB
#                 rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
#                 # Process the frame with MediaPipe Holistic
#                 results = holistic.process(rgb_frame)
#                 print(results)
                
#                 # Draw landmarks on the frame
#                 if results.pose_landmarks:
#                     # Draw pose landmarks
#                     mp_holistic.draw_landmarks(frame2, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    
#                     # Draw left hand landmarks
#                     mp_holistic.draw_landmarks(frame2, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    
#                     # Draw right hand landmarks
#                     mp_holistic.draw_landmarks(frame2, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

#                 # Encode the frame as JPEG
#                 ret, buffer = cv2.imencode('.jpg', frame2)
#                 frame2 = buffer.tobytes()
                
#                 yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')



###################################################################################################################
###                 Working code dont delete
# def sign():
#     global camera2
#     camera2 = cv2.VideoCapture(0)

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             try:
#                 success, frame = camera2.read()
#                 if not success:
#                     break

#                 # Process the frame with Holistic
#                 image, results = mediapipe_detection(frame, holistic)

#                 # Draw landmarks on the frame
#                 draw_styled_landmarks(image, results)

#                 # Encode the frame as JPEG
#                 _, buffer = cv2.imencode('.jpg', image)
#                 frame_data = buffer.tobytes()

#                 # Yield the frame data as a response
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
#             except Exception as e:
#                 print(f"Error: {str(e)}")

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def draw_styled_landmarks(image, results):
#     mp_drawing = mp.solutions.drawing_utils
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
#                               mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#                               mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
########################################################################################################################

# Function for processing video frames and making predictions
### code2
# def sign():
#     global camera2
#     camera2 = cv2.VideoCapture(0)

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             try:
#                 success, frame = camera2.read()
#                 if not success:
#                     break

#                 # Process the frame with Holistic
#                 image, results = mediapipe_detection(frame, holistic)

#                 # Extract hand landmarks
#                 hand_landmarks = extract_hand_landmarks(results)

#                 # Make predictions using your model
#                 recognized_sign = predict_sign(hand_landmarks)

#                 # Display the recognized sign on the frame
#                 cv2.putText(image, recognized_sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#                 # Encode the frame as JPEG
#                 _, buffer = cv2.imencode('.jpg', image)
#                 frame_data = buffer.tobytes()

#                 # Yield the frame data as a response
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
#             except Exception as e:
#                 print(f"Error: {str(e)}")

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# # def extract_hand_landmarks(results):
# #     if results.right_hand_landmarks:
# #         hand_landmarks = results.right_hand_landmarks.landmark
# #     elif results.left_hand_landmarks:
# #         hand_landmarks = results.left_hand_landmarks.landmark
# #     else:
# #         hand_landmarks = []

# #     # Extract x, y, and z coordinates of hand landmarks
# #     landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks])
# #     return landmarks
# def extract_hand_landmarks(results):
#     if results.right_hand_landmarks:
#         hand_landmarks = results.right_hand_landmarks.landmark
#     elif results.left_hand_landmarks:
#         hand_landmarks = results.left_hand_landmarks.landmark
#     else:
#         hand_landmarks = []

#     # Extract x, y, and z coordinates of hand landmarks
#     landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks])

#     # Ensure landmarks have at least 50 frames
#     landmarks = np.tile(landmarks, (50, 1))

#     # Reshape to (None, 50, 1662)
#     landmarks = landmarks.reshape((1, 50, 1662))

#     return landmarks


# def preprocess_hand_landmarks(hand_landmarks):
#     # Implement any additional preprocessing steps here if needed
#     return hand_landmarks

# # def predict_sign(hand_landmarks):
# #     hand_landmarks = preprocess_hand_landmarks(hand_landmarks)
    
# #     # Assuming hand_landmarks is a 1D array, you might need to reshape it to match the expected input shape
# #     hand_landmarks = hand_landmarks.reshape((1, 50, 1662))  # Adjust the shape as needed
    
# #     # Make predictions using your model
# #     res = model.predict(hand_landmarks)[0]
# #     recognized_sign = actions[np.argmax(res)]

# #     return recognized_sign
# def predict_sign(hand_landmarks):
#     hand_landmarks = preprocess_hand_landmarks(hand_landmarks)

#     # Make predictions using your model
#     res = model.predict(hand_landmarks)[0]
#     recognized_sign = actions[np.argmax(res)]

#     return recognized_sign

###############################################################################
model = keras.models.load_model('gru.h5')  # Replace with your model path
actions = ['go', 'home', 'I', 'water']
# label_map = {label:num for num, label in enumerate(actions)}

# sequences, labels = [], []
# for action in actions:
#     for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])
sequence = []
sentence = [] 
predictions = []
threshold = 0.7

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
# def sign():
#     global camera2
#     camera2 = cv2.VideoCapture(0)

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             try:
#                 success, frame = camera2.read()
#                 if not success:
#                     break

#                 # Process the frame with Holistic
#                 image, results = mediapipe_detection(frame, holistic)

#                 # Extract keypoints from hand landmarks
#                 hand_keypoints = extract_keypoints(results)
#                 # print(f"Hand Keypoints Shape: {hand_keypoints.shape}")

#                 # Make predictions using your model
#                 preprocessed_hand_keypoints = preprocess_hand_keypoints(hand_keypoints)
#                 recognized_sign, confidence = predict_sign(preprocessed_hand_keypoints)

#                 # Display the recognized sign on the frame
#                 cv2.putText(image, f"{recognized_sign} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#                 # Draw landmarks on the frame
#                 draw_styled_landmarks(image, results)

#                 # Encode the frame as JPEG
#                 _, buffer = cv2.imencode('.jpg', image)
#                 frame_data = buffer.tobytes()

#                 # Yield the frame data as a response
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
#             except Exception as e:
#                 print(f"Error: {str(e)}")

#######################################################################################################
detected = ""
def sign():
    global camera2
    camera2 = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            try:
                success, frame = camera2.read()
                if not success:
                    break

                # Process the frame with Holistic
                image, results = mediapipe_detection(frame, holistic)

                #draw landmarks
                draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                global sequence
                global predictions
                global sentence
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    global detected
                    detected = actions[np.argmax(res)]
                    predictions.append(np.argmax(res))
                    
                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                # # Viz probabilities
                # image = prob_viz(res, actions, image, colors)
                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, detected, (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                

                # Encode the frame as JPEG
                _, buffer = cv2.imencode('.jpg', image)
                frame_data = buffer.tobytes()

                # Yield the frame data as a response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            except Exception as e:
                print(f"Error: {str(e)}")


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def preprocess_hand_keypoints(hand_keypoints):
    print("Before Preprocessing:", hand_keypoints.shape)
    # Assuming hand_keypoints has shape (1, 50, 3) and you want to reshape it to (1, 50, 1662)
    num_landmarks = 50
    keypoints_dim = 1662  # Adjust this value based on the actual dimension

    # Pad with zeros or truncate based on the number of keypoints
    if hand_keypoints.shape[1] < num_landmarks:
        # Pad with zeros if there are fewer than 50 landmarks
        hand_keypoints = np.pad(hand_keypoints, ((0, 0), (0, num_landmarks - hand_keypoints.shape[1]), (0, 0)), 'constant', constant_values=0)
    elif hand_keypoints.shape[1] > num_landmarks:
        # Take the first 50 landmarks if there are more than 50
        hand_keypoints = hand_keypoints[:, :num_landmarks, :]

    # Ensure the resulting array has the correct shape
    hand_keypoints = hand_keypoints.reshape((1, num_landmarks, keypoints_dim))
    print("After Preprocessing:", hand_keypoints.shape) 

    return hand_keypoints


# def predict_sign(preprocessed_hand_keypoints):
#     # Check if hand keypoints are available for prediction
#     if preprocessed_hand_keypoints.shape == (1, 50, 1662):
#         # Make predictions using your model
#         res = model.predict(preprocessed_hand_keypoints)[0]
#         recognized_sign = actions[np.argmax(res)]
#         confidence = res[np.argmax(res)]
        
#         return recognized_sign, confidence
#     else:
#         return "No hand keypoints", 0

def predict_sign(hand_keypoints):
    hand_keypoints = preprocess_hand_keypoints(hand_keypoints)
    print("Hand Keypoints:", hand_keypoints)

    # Check if hand keypoints are available for prediction
    if len(hand_keypoints) > 0:
        hand_keypoints = np.squeeze(hand_keypoints, axis=0)  # Remove the batch dimension

        # Make predictions using your model
        res = model.predict(hand_keypoints[np.newaxis, ...])[0]
        recognized_sign = actions[np.argmax(res)]
        confidence = res[np.argmax(res)]
    else:
        recognized_sign = "No hand keypoints"
        confidence = 0.0

    return recognized_sign, confidence

##################################################################################################################


# def sign():
#     global camera2
#     camera2 = cv2.VideoCapture(0)

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             try:
#                 success, frame = camera2.read()
#                 if not success:
#                     break

#                 # Process the frame with Holistic
#                 image, results = mediapipe_detection(frame, holistic)

#                 # Extract hand keypoints
#                 hand_keypoints = extract_keypoints(results)

#                 # Make predictions using your model
#                 recognized_sign, confidence = predict_sign(hand_keypoints)

#                 # Display the recognized sign and confidence on the frame
#                 text = f"{recognized_sign} ({confidence:.2f})"
#                 cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#                 # Draw landmarks on the frame for visualization
#                 draw_styled_landmarks(image, results)

#                 # Encode the frame as JPEG
#                 _, buffer = cv2.imencode('.jpg', image)
#                 frame_data = buffer.tobytes()

#                 # Yield the frame data as a response
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
#             except Exception as e:
#                 print(f"Error: {str(e)}")

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results

# def extract_keypoints(results):
#     if results is not None:
#         pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#         face_landmarks = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#         left_hand_landmarks = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#         right_hand_landmarks = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

#         # Concatenate the landmarks
#         all_landmarks = np.concatenate([pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks])

#         # Reshape to the desired shape (50, 1662)
#         num_landmarks = 30
#         landmarks_dim = 1662  # Adjust this value based on the actual dimension

#         # Pad with zeros or truncate based on the number of landmarks
#         if len(all_landmarks) < num_landmarks * landmarks_dim:
#             # Pad with zeros if there are fewer than 50 landmarks
#             all_landmarks = np.pad(all_landmarks, (0, num_landmarks * landmarks_dim - len(all_landmarks)), 'constant', constant_values=0)
#         elif len(all_landmarks) > num_landmarks * landmarks_dim:
#             # Take the first 50 landmarks if there are more than 50
#             all_landmarks = all_landmarks[:num_landmarks * landmarks_dim]

#         # Ensure the resulting array has the correct shape
#         all_landmarks = all_landmarks.reshape((1, num_landmarks, landmarks_dim))

#         print(f"Extracted Keypoints Shape: {all_landmarks.shape}")

#         return all_landmarks
#     else:
#         return np.zeros((1, 50, 1662))


# def preprocess_hand_keypoints(hand_keypoints):
#     print("Before Preprocessing:", hand_keypoints.shape)
#     # Assuming hand_keypoints has shape (1, 30, 3) and you want to reshape it to (1, 30, 1662)
#     num_landmarks = 30
#     keypoints_dim = 1662  # Adjust this value based on the actual dimension

#     # Pad with zeros or truncate based on the number of keypoints
#     if hand_keypoints.shape[1] < num_landmarks:
#         # Pad with zeros if there are fewer than 30 landmarks
#         hand_keypoints = np.pad(hand_keypoints, ((0, 0), (0, num_landmarks - hand_keypoints.shape[1]), (0, 0)), 'constant', constant_values=0)
#     elif hand_keypoints.shape[1] > num_landmarks:
#         # Take the first 30 landmarks if there are more than 30
#         hand_keypoints = hand_keypoints[:, :num_landmarks, :]

#     # Ensure the resulting array has the correct shape
#     hand_keypoints = hand_keypoints.reshape((1, num_landmarks, keypoints_dim))
#     print("After Preprocessing:", hand_keypoints.shape) 

#     return hand_keypoints

# def predict_sign(hand_keypoints):
#     hand_keypoints = preprocess_hand_keypoints(hand_keypoints)
#     print("Hand Keypoints:", hand_keypoints)

#     # Check if hand keypoints are available for prediction
#     if len(hand_keypoints) > 0:
#         hand_keypoints = np.squeeze(hand_keypoints, axis=0)  # Remove the batch dimension

#         # Make predictions using your model
#         res = model

# def draw_styled_landmarks(image, results):
#     mp_drawing = mp.solutions.drawing_utils
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
#                               mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#                               mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))








###                                                          till here
################################################################################################################################################

### Starting of routes ###

@app.route('/')
def index():
    # if type=="face":
    return render_template('index.html')
    # elif type=="sign_lang":
    #     return render_template('index_sign.html')

@app.route('/login')
def login():
    # Query to retrieve image data, usernames, user_ids, and user_data for all users
    with app.app_context():
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT image_data, name, user_id, user_data FROM users")
        user_data_list = cursor.fetchall()
        cursor.close()

    # Process each user data
    for user_data in user_data_list:
        image_data = user_data['image_data']
        username = user_data['name']
        user_id = user_data['user_id']
        user_data_info = user_data['user_data']

        person_image = face_recognition.load_image_file(io.BytesIO(image_data))  # Convert bytes to image
        person_face_encoding = face_recognition.face_encodings(person_image)[0]  # Assuming only one face per image

        global known_face_encodings
        known_face_encodings.append(person_face_encoding)
        global known_face_names
        known_face_names.append(username)
        global known_user_ids
        known_user_ids.append(user_id)
        global known_user_data
        known_user_data.append(user_data_info)
    # Add code to render the login page
    return render_template('login.html', show_warning=False)

@app.route('/registration')
def registration():
    # Add code to render the registration page
    return render_template('registration.html')

################################################################################################################################################

## Video Feeds URl ###

@app.route('/video')
def video():
    if type=="sign_lang":
        hello = 1
        return Response(sign(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif type=="face":
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        ##start capturing sign language, and open camera for that 
        #return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    return Response(sign(), mimetype='multipart/x-mixed-replace; boundary=frame')

################################################################################################################################################

### Login Handler ###
@app.route('/submit',methods=['POST','GET'])
def submit():
    get_name=""
    if request.method=='POST':
        get_name=request.form['yourname']
    res=''

    if get_name==fetch_name and fetch_name != "Unknown":
        res='success'
    else:
        res="fail"
    return redirect(url_for(res,urname=get_name))

@app.route('/success/<string:urname>')
def success(urname):
    global camera
    camera=""
    global type
    type = "sign_lang"
    res=urname
    data = "name , Age, Background info"
    #fetch user data based on name and store in variable data    
    return render_template("success.html", result=res, result2=type, name=known_face_names[data_index], userid=known_user_ids[data_index], data=known_user_data[data_index],)
    # return render_template('success.html',result=res,data="user-info")

@app.route('/fail/<string:urname>')
def fail(urname):
    res = urname
    return render_template('login.html', show_warning=True, result=res)

#########################################################################################################################################

### Registration Handler

# File Upload Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    user_id = request.form['user_id']
    password = request.form['password']
    email = request.form['email']
    user_data = request.form['user_data']

    # Handle file upload and image data
    if 'image' in request.files:
        image = request.files['image']
        if image.filename != '' and allowed_file(image.filename):
            image_data = image.read()  # Read the binary image data
        else:
            image_data = None
    else:
        image_data = None

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO users (name, user_id, password, email, user_data, image_data) VALUES (%s, %s, %s, %s, %s, %s)",
                (name, user_id, password, email, user_data, image_data))
    mysql.connection.commit()
    cur.close()

    return redirect(url_for('index'))

#############################################################################################################################################
#Translation module 3
@app.route('/translate', methods=['POST'])
def translate():
    # global camera2
    # camera2 = ""

    lang = request.form['language']

    def load_glove_vectors(file_path):
        glove_vectors = {}
        with open(file_path, encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                glove_vectors[word] = vector
        return glove_vectors
    
    # def calculate_sentence_embedding(words, word_vectors):
    #     word_embeddings = [word_vectors[word] for word in words if word in word_vectors]
    #     if word_embeddings:
    #         sentence_embedding = np.mean(word_embeddings, axis=0)
    #         return sentence_embedding
    #     else:
    #         return None
        
    sentences = [
    "i like to walk in the park",
    "i drink coffee everyday",
    "i like dogs",
    "hello dear",
    "i read a book yesterday",
    "can you please get me some water",
    "i want my medicine",
    "i am going to my home",
    "i am going to market",
    "i am going to my home in the evening",
    ]

    glove_file_path = 'glove.6B.100d.txt'  # Adjust the file path to your GloVe file
    glove_vectors = load_glove_vectors(glove_file_path)

    input_tokens = sentence  # tokens from module 2 come into this variable
    # flag = "Yes"  # make an option in the beginning to ask for Hindi language. It can be a bool variable
    # flag = flag.lower()

    word_embeddings = [glove_vectors[word] for word in input_tokens if word in glove_vectors]
    closest_sentence = ""
    if word_embeddings:
        input_embedding = np.mean(word_embeddings, axis=0)
        input_embedding_reshaped = input_embedding.reshape(1, -1)

        similarities = [cosine_similarity(input_embedding_reshaped, [np.mean([glove_vectors[word] for word in sentence.split() if word in glove_vectors], axis=0)])[0][0] for sentence in sentences]

        closest_sentence_index = np.argmax(similarities)
        closest_sentence = sentences[closest_sentence_index]
    else:
        print("No valid word embeddings found in input.")


    if closest_sentence:
        if lang == 'hi':
            closest_sentence = EngtoHindi(closest_sentence)
            print(closest_sentence.convert)
            tts = gTTS(closest_sentence.convert, lang="hi")
        else:
            tts = gTTS(closest_sentence, lang='en', tld='co.in')
            print(closest_sentence)
    else:
        tts = gTTS("Sorry, I cannot read that to you.")

    word = closest_sentence

    audio_filename = 'output_audio.mp3'  # Specify the audio file name
    audio_path = os.path.join(app.root_path, audio_filename)  # Save the audio file in the root directory
    tts.save(audio_path)

    # Play audio using pygame
    # pygame.mixer.init()
    # pygame.mixer.music.load(audio_path)
    # pygame.mixer.music.play()

    # Wait for the audio to finish playing
    # while pygame.mixer.music.get_busy():
    #     pygame.time.Clock().tick(10)

    return render_template('translate.html', word=word, audio_path = audio_filename)

@app.route('/audio/<filename>')
def audio(filename):
    return send_file(filename, mimetype='audio/mpeg')

#############################################################################################################################################

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)


# @app.route('/translate',methods=['POST','GET'])
# def translate():
#     model=""
#     if request.method=='POST':
#         model=request.form['language']
#     ur='sign'
#     return redirect(url_for(ur, ans=model))


# @app.route('/sign/<string:ans>')
# def translate(ans):
#     return render_template("fail.html", result=type)
    




    