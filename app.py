from flask import Flask, render_template, redirect, request, url_for, Response, jsonify, session
from flask_cors import CORS, cross_origin
import logging
import time
import pandas as pd
import os
from werkzeug.utils import secure_filename
from src.FaceTime import FaceIdentity
from src.image_to_face_embeddings import FaceEmbeddingPipeline
from src import config

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch")

logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Set configuration
app.config['RegistrationData'] = config.feature_embs
os.makedirs(app.config['RegistrationData'], exist_ok=True)

trained_data = config.Trained_data_dir
do_training_data = config.unTrained_data_dir

# Create an embedding instance
embeddings = FaceEmbeddingPipeline(trained_data, do_training_data)
logging.info("Creating embedding instance for input data...")

getEmotions = False
getFaceLoc = True
camera_on = False
faceRec = None

# Load user data
def load_user_data():
    csvpath = os.path.join(app.config['RegistrationData'], "UserData.csv")
    if os.path.exists(csvpath):
        logging.info(f"{csvpath} already exists...")
        return pd.read_csv(csvpath)
    else:
        logging.info(f"{csvpath} created...")
        return pd.DataFrame(columns=['firstname', 'lastname', 'email', 'password', 'images'])

UserData = load_user_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_email', methods=['POST'])
@cross_origin()
def check_email():
    data = request.get_json()
    email = data.get('email')

    csvpath = os.path.join(app.config['RegistrationData'], "UserData.csv")

    if not os.path.exists(csvpath):
        return jsonify({'status': 'success'})

    global UserData
    UserData = pd.read_csv(csvpath)

    if email in UserData['email'].values:
        return jsonify({'status': 'error', 'message': 'Email already registered.'})

    return jsonify({'status': 'success'})

@app.route('/register', methods=['POST'])
@cross_origin()
def register():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    email = request.form['email']
    password = request.form['password']

    global UserData

    if email in UserData['email'].values:
        logging.info(f"{email} Already registered")
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

    return jsonify({'status': 'success', 'message': 'Registration Complete, please LogIn...'})

@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    csvpath = os.path.join(app.config['RegistrationData'], "UserData.csv")

    if os.path.exists(csvpath):
        global UserData
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
@cross_origin()
def update_settings():
    data = request.get_json()
    global getEmotions, getFaceLoc
    getEmotions = data.get('getEmotions', getEmotions)
    getFaceLoc = data.get('getFaceLocation', getFaceLoc)
    return jsonify({'status': 'success'})

def get_frames():
    global camera_on, faceRec

    while camera_on:
        frame = faceRec.RealTimeRecognition(getEmotions=getEmotions, getFaceLoc=getFaceLoc)
        logging.debug(f"Type of frame before conversion: {type(frame)}")

        if isinstance(frame, str):
            logging.debug(f"Frame content before conversion: {frame[:100]}")
            frame = frame.encode('utf-8')
        logging.debug(f"Type of frame after conversion to bytes: {type(frame)}")

        if not isinstance(frame, bytes):
            raise TypeError(f"Frame is not in bytes. Frame type: {type(frame)}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
@cross_origin()
def start_camera():
    global camera_on, faceRec

    if not camera_on:
        faceRec = FaceIdentity(videoPath=0)
        camera_on = True

    return jsonify({'status': 'success'})

@app.route('/stop_camera', methods=['POST'])
@cross_origin()
def stop_camera():
    global camera_on, faceRec

    camera_on = False
    faceRec = None
    return jsonify({'status': 'success'})

@app.route('/upload_dataset', methods=['POST'])
@cross_origin()
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

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
    # app.run()
    
