from flask import Flask, render_template, redirect, request, url_for, Response, jsonify, session
from flask_cors import CORS, cross_origin
import logging
import cv2
from src.FaceTime import FaceIdentity
import time
import pandas as pd
from src import config
import os
from werkzeug.utils import secure_filename
from src.image_to_face_embeddings import FaceEmbeddingPipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch")

logging.basicConfig(level=logging.DEBUG)

# Create a Flask app instance at the module level
app = Flask(__name__)

# Initialize CORS with default settings (allow all origins)
CORS(app)

os.makedirs(config.feature_embs, exist_ok=True)
app.config['RegistrationData'] = config.feature_embs

# Define other configurations and routes here
trained_data = config.Trained_data_dir
do_training_data = config.unTrained_data_dir

# 2. create an embedding instance
embeddings = FaceEmbeddingPipeline(trained_data, do_training_data)
logging.info("Creating embedding instance for input data...")

# get face emotions and face locations
getEmotions = False
getFaceLoc = True

UserData = pd.DataFrame(columns=['firstname', 'lastname', 'email', 'password', 'images'])
faceRec = None
camera_on = False

# Define your routes here
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_email', methods=['POST'])
def check_email():
    data = request.get_json()
    email = data.get('email')
    
    csvpath = os.path.join(app.config['RegistrationData'], "UserData.csv")
    if not os.path.exists(csvpath):
        return jsonify({'status': 'success'})

    UserData = pd.read_csv(csvpath)
    if email in UserData['email'].values:
        return jsonify({'status': 'error', 'message': 'Email already registered.'})

    return jsonify({'status': 'success'})

@app.route('/register', methods=['POST'])
def register():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    email = request.form['email']
    password = request.form['password']
    
    csvpath = os.path.join(app.config['RegistrationData'], "UserData.csv")
    if not os.path.exists(csvpath):
        return jsonify({'status': 'success'})

    UserData = pd.read_csv(csvpath)
    
    if email in UserData['email'].values:
        logging.info(f"{email} Already ")
        return jsonify({'status': 'error', 'message': 'Email already registered'})

    saveImagepath = os.path.join(do_training_data, "_".join([firstname, lastname]))
    os.makedirs(saveImagepath, exist_ok=True)

    image_paths = []
    for i in range(10):
        image = request.files.get(f'image{i}')
        if image:
            filename = secure_filename(f"{firstname}_{i}.jpg")
            image_path = os.path.join(saveImagepath, filename)
            image.save(image_path)
            image_paths.append(image_path)

    new_user_data = pd.DataFrame({
        'firstname': [firstname],
        'lastname': [lastname],
        'email': [email],
        'password': [password],
        'images': [image_paths]
    })

    UserData = pd.concat([UserData, new_user_data], ignore_index=True)
    pathToSaveData = os.path.join(app.config['RegistrationData'], "UserData.csv")
    UserData.to_csv(pathToSaveData, mode='a', index=False, header=not os.path.exists(pathToSaveData))

    embeddings.process_and_embed_faces()
    return jsonify({'status': 'success', 'message': 'Registration Complete please LogIn...'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    csvpath = os.path.join(app.config['RegistrationData'], "UserData.csv")

    if os.path.exists(csvpath):
        UserData = pd.read_csv(csvpath)
        user = UserData[(UserData['email'] == email) & (UserData['password'].astype(str) == password)]
        if not user.empty:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid email or password.'})
    else:
        return jsonify({'status': 'error', 'message': 'User data not found.'})

@app.route('/face_recognition_results')
def face_recognition_results():
    return render_template('face_recognition_results.html')

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    global getEmotions, getFaceLoc
    getEmotions = data.get('getEmotions', getEmotions)
    getFaceLoc = data.get('getFaceLocation', getFaceLoc)
    return jsonify({'status': 'success'})

@app.route('/video_feed')
def video_feed():
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global faceRec, camera_on
    if not camera_on:
        faceRec = FaceIdentity(videoPath=0)
        camera_on = True
    return jsonify({'status': 'success'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_on, faceRec
    camera_on = False
    faceRec = None
    return jsonify({'status': 'success'})

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'files[]' not in request.files:
        return jsonify({'status': 'error', 'message': 'No files uploaded'}), 400

    files = request.files.getlist('files[]')
    for file in files:
        if file and file.filename:
            directory = os.path.dirname(file.filename)
            os.makedirs(os.path.join(do_training_data, directory), exist_ok=True)
            file_path = os.path.join(do_training_data, file.filename)
            file.save(file_path)

    embeddings.process_and_embed_faces()
    return jsonify({'status': 'success', 'message': 'Dataset uploaded successfully'})

def get_frames():
    global camera_on, faceRec, getEmotions, getFaceLoc
    while camera_on:
        frame = faceRec.RealTimeRecognition(getEmotions=getEmotions, getFaceLoc=getFaceLoc)


        # Log the type and content of frame
        logging.debug(f"Type of frame before conversion: {type(frame)}")

        if isinstance(frame, str):
            logging.debug(f"Frame content before conversion: {frame[:100]}")  # Log first 100 chars if str
            frame = frame.encode('utf-8')
        logging.debug(f"Type of frame after conversion to bytes: {type(frame)}")

        if not isinstance(frame, bytes):
            raise TypeError(f"Frame is not in bytes. Frame type: {type(frame)}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.1)


if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=logging.INFO)
    # logging.info("Starting server with Hypercorn...")
    # from hypercorn.asyncio import serve
    # from hypercorn.config import Config
    # import asyncio

    # config = Config()
    # config.bind = ["0.0.0.0:8000"]
    # asyncio.run(serve(app, config))

    app.run()
    
    
    # app.run(debug=True, host='0.0.0.0', port=8000)
    # THIS APP WILL RUN ON 127.0.0.1:8000