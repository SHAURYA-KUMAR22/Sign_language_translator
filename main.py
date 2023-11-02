from flask import Flask, render_template, Response,url_for,redirect,request
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
from mediapipe.solutions import Holistic

app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera2 = ""

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

# List to store face encodings and names
known_face_encodings = []
known_face_names = ["shaurya"]

# Query to retrieve image data for all users
with app.app_context():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT image_data FROM users")
    image_data_list = cursor.fetchall()
    cursor.close()

# Process each image data
for image_data in image_data_list:
    person_image = face_recognition.load_image_file(io.BytesIO(image_data['image_data']))  # Convert bytes to image
    person_face_encoding = face_recognition.face_encodings(person_image)[0]  # Assuming only one face per image

    known_face_encodings.append(person_face_encoding)

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
            

def sign():
    global camera2
    camera2 = cv2.VideoCapture(0)
    while True:
        success, frame2 = camera2.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame2, (0, 0), fx=0.25, fy=0.25)
            ret, buffer = cv2.imencode('.jpg', frame2)
            frame2 = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

# # Load your pre-trained model
# model = keras.models.load_model('action.h5')  # Replace with your model path

# # Define actions (replace with your actual action names)
# actions = ["action1", "action2", "action3"]

# Function for processing video frames and making predictions
# def sign():
#     global camera2
#     camera2 = cv2.VideoCapture(0)

#     with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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
#                 frame2 = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
#             except Exception as e:
#                 print(f"Error: {str(e)}")

# def preprocess_frame(frame):
#     # Implement your frame preprocessing here (e.g., resizing, converting to RGB)
#     processed_frame = cv2.resize(frame, (224, 224))
#     processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
#     processed_frame = processed_frame / 255.0  # Normalize the frame

#     return processed_frame

# def predict_action(frame):
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         # Read feed
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = holistic.process(image)

#         # Check if there are pose landmarks
#         if results.pose_landmarks:
#             keypoints = extract_keypoints(results)
#             keypoints = np.expand_dims(keypoints, axis=0)  # Add batch dimension

#             # Make predictions using your model
#             res = model.predict(keypoints)[0]
#             recognized_action = actions[np.argmax(res)]
#         else:
#             recognized_action = "No pose detected"

#     return recognized_action

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
#     return np.concatenate([pose, face, lh, rh])
            
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
    return render_template("success.html", result=res, result2=type, data=data)
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
    




    