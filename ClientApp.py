from flask import Flask, render_template, redirect, request, url_for, Response, jsonify, session
from flask_cors import CORS, cross_origin
import logging

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


class FaceRecognitionApp:
    def __init__(self):
        self.app = Flask(__name__)
        
        # Initialize CORS with default settings (allow all origins)
        CORS(self.app)

        os.makedirs(config.feature_embs, exist_ok=True)
        self.app.config['RegistrationData'] = config.feature_embs

        

        self.trained_data = config.Trained_data_dir
        self.do_training_data = config.unTrained_data_dir

        # 2. create an embedding instance
        self.embeddings = FaceEmbeddingPipeline(self.trained_data, self.do_training_data)
        logging.info("Creating embedding instance for input data...")

        # get face emotions and face locations
        self.getEmotions = False
        self.getFaceLoc = True

        self.UserData = self.load_user_data()
        self.faceRec = None
        self.camera_on = False
        self.setup_routes()
    
    def load_user_data(self):
        csvpath = os.path.join(self.app.config['RegistrationData'], "UserData.csv")
        if os.path.exists(csvpath):
            UserData = pd.read_csv(csvpath)
            logging.info(f"{csvpath} already exists...")

        else:
            UserData = pd.DataFrame(columns=['firstname', 'lastname', 'email', 'password', 'images'])
            logging.info(f"{csvpath} created...")

        return UserData

    def setup_routes(self):
        self.app.add_url_rule('/', 'home', self.home)
        self.app.add_url_rule('/check_email', 'check_email', self.check_email, methods=['POST'])
        self.app.add_url_rule('/register', 'register', self.register, methods=['POST'])
        self.app.add_url_rule('/login', 'login', self.login, methods=['POST'])
        self.app.add_url_rule('/face_recognition_results', 'face_recognition_results', self.face_recognition_results)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/start_camera', 'start_camera', self.start_camera, methods=['POST'])
        self.app.add_url_rule('/stop_camera', 'stop_camera', self.stop_camera, methods=['POST'])
        self.app.add_url_rule('/upload_dataset', 'upload_dataset', self.upload_dataset, methods=['POST'])  # New route
        self.app.add_url_rule('/update_settings', 'update_settings', self.update_settings, methods=['POST'])  # New route

    def home(self):
        return render_template('index.html')

    def check_email(self):
        data = request.get_json()
        email = data.get('email')

        csvpath = os.path.join(self.app.config['RegistrationData'], "UserData.csv")

        if not os.path.exists(csvpath):
            return jsonify({'status': 'success'})

        self.UserData = pd.read_csv(csvpath)

        if email in self.UserData['email'].values:
            return jsonify({'status': 'error', 'message': 'Email already registered.'})

        return jsonify({'status': 'success'})

    def register(self):
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        password = request.form['password']

        if email in self.UserData['email'].values:
            logging.info(f"{email} Already ")
            return jsonify({'status': 'error', 'message': 'Email already registered'})

        saveImagepath = os.path.join(self.do_training_data, "_".join([firstname, lastname]))
        os.makedirs(saveImagepath, exist_ok=True)

        # Collect the images and their paths
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

        self.UserData = pd.concat([self.UserData, new_user_data], ignore_index=True)
        pathToSaveData = os.path.join(self.app.config['RegistrationData'], "UserData.csv")
        self.UserData.to_csv(pathToSaveData, mode='a', index=False, header=not os.path.exists(pathToSaveData))

        # Create embeddings of captured images
        self.embeddings.process_and_embed_faces()

        return jsonify({'status': 'success', 'message': 'Registration Complete please LogIn...'})

    def login(self):
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        csvpath = os.path.join(self.app.config['RegistrationData'], "UserData.csv")

        if os.path.exists(csvpath):
            self.UserData = pd.read_csv(csvpath)

            user = self.UserData[(self.UserData['email'] == email) & (self.UserData['password'].astype(str) == password)]

            if not user.empty:
                return jsonify({'status': 'success'})
            else:
                return jsonify({'status': 'error', 'message': 'Invalid email or password.'})
        else:
            return jsonify({'status': 'error', 'message': 'User data not found.'})

    def face_recognition_results(self):
        return render_template('face_recognition_results.html')

    def update_settings(self):
        data = request.get_json()
        self.getEmotions = data.get('getEmotions', self.getEmotions)
        self.getFaceLoc = data.get('getFaceLocation', self.getFaceLoc)
        return jsonify({'status': 'success'})

    def get_frames(self):
        print("Running...Get Frames")

        while self.camera_on:
            frame = self.faceRec.RealTimeRecognition(getEmotions=self.getEmotions, getFaceLoc=self.getFaceLoc)

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

    def video_feed(self):
        return Response(self.get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def start_camera(self):
        if not self.camera_on:
            self.faceRec = FaceIdentity(videoPath=0)
            self.camera_on = True

        return jsonify({'status': 'success'})

    def stop_camera(self):
        self.camera_on = False
        self.faceRec = None
        return jsonify({'status': 'success'})

    def upload_dataset(self):
        if 'files[]' not in request.files:
            return jsonify({'status': 'error', 'message': 'No files uploaded'}), 400

        files = request.files.getlist('files[]')
        for file in files:
            if file and file.filename:
                # Ensure the directory structure is created
                directory = os.path.dirname(file.filename)
                os.makedirs(os.path.join(self.do_training_data, directory), exist_ok=True)

                # Save the file
                file_path = os.path.join(self.do_training_data, file.filename)
                file.save(file_path)

        # After saving, you might want to update embeddings
        self.embeddings.process_and_embed_faces()

        return jsonify({'status': 'success', 'message': 'Dataset uploaded successfully'})
    
    def run(self):
        # self.app.run(debug=True, host='0.0.0.0', port=8000)
        # self.app.run(host='0.0.0.0')
        
        
        logging.info("Starting server with Hypercorn...")
        from hypercorn.asyncio import serve
        from hypercorn.config import Config
        import asyncio
        config = Config()
        config.bind = ["0.0.0.0:8000"]
        asyncio.run(serve(self.app, config))


if __name__ == '__main__':
    app = FaceRecognitionApp()
    app.run()
    
    
Faceapp = FaceRecognitionApp().app  # Expose the app instance for Hypercorn
