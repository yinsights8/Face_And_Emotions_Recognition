# Face_And_Emotions_Recognition

Our face recognition web app leverages the powerful SCRFD face detector from the InsightFace repository, incorporating advanced Emotion Detection for a comprehensive user experience. The models, developed using PyTorch, ensure high accuracy and performance, with the SCRFD face detector implemented as an ONNX model for efficient execution. ArcFace is chosen for its superior capability in achieving high recognition accuracy through additive angular margin loss, making it a top choice compared to other models. This combination of technologies provides a robust, responsive, and accurate face recognition system, suitable for various applications, from security to user engagement.

**1. Technologies Used**

   - **Face Detection**
   - **Face Recognition**
   - **Face Tracking**
   - **Matching Algorithm**
   - **Emotion Detection**


**1. Face Detector**

   - **SCRFD from Insight Face**:
     - SCRFD (Single-Shot Scale-Aware Face Detector) is designed for real-time face detection across various scales. It is particularly effective in detecting faces at different resolutions within the same image.
   - **Yolovv8-face**:
     - Yolovv8-face is based on the YOLO (You Only Look Once) architecture, specializing in face detection. It provides real-time face detection with a focus on efficiency and accuracy.
   - **Retinaface**:
     - Retinaface is a powerful face detection algorithm known for its accuracy and speed. It utilizes a single deep convolutional network to detect faces in an image with high precision.


**2. Face Recognition**

   1. **ArchFace**:
      ArcFace is a state-of-the-art face recognition algorithm that focuses on learning highly discriminative features for face verification and identification. It is known for its robustness to variations in lighting, pose, and facial expressions.
      ![image](https://github.com/yinsights8/Face_And_Emotions_Recognition/blob/main/static/images/search_identities.png?raw=true)
      


**3. Face Tracking**

   1. **Sort**:
      SORT (Simple Online and Realtime Tracking) is a tracking algorithm that uses a Kalman filter and the Hungarian algorithm to efficiently track objects in video frames by estimating their positions and matching them across frames.


**4.Matching Algorithm**

   1. **Cosine Similariy**:
         Cosine similarity measures the cosine of the angle between two vectors, representing the feature embeddings of faces. In the context of face matching, it quantifies how similar the faces are by comparing their feature vectors, with a cosine similarity close to 1 indicating high similarity and a value close to 0 indicating low similarity.

      ![image](https://github.com/yinsights8/Face_And_Emotions_Recognition/blob/main/static/images/Cosine_Similarity_in_Face_recognition.png?raw=true)

**4. Emotion Detection**
   1. *approach 1.* using Mediapipe facemesh landmarks
   2. *approach 2.* This project aims to classify the emotion on a person's face into one of seven categories, using deep convolutional neural networks. The model is trained on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

This project Uses *approach 2.*

## Here are the step-by-step instructions for running the code:

### 1. Create a new environment with Python 3.9 
```
## for in VS code
pip install virtualenv
python -m venv venv        # 'venv' is the name of the virtual environment
venv\Scripts\activate

## for Conda
conda create -n face-dev python==3.9 -y
conda activate face-dev
```

### 2. Install PyTorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### 3. Install requirements.txt:
```
pip install -r requirements.txt
```

### 4. Navigate to the Code Directory:
```
cd path/to/your/code/directory
```
### 5. Run the Python Script for face recognition locally :
```
python main.py
```
here enter your first name and last name then it will click some live images to show some predictions 

## 6. Run the Python Script as a web app:
```
python ClientApp.py
```
Here sign up after signing up wait for 10 secends. it will click images and then automatically log in. Then start the webcam and wait for 5 - 10 sec. it will start live recognition


## On the web app you can also add images from your end but the folder structure should match
```
      ├── person_name1
      │   └── image1.jpg
      │   └── image2.jpg
      └── person_name2
          └── image1.jpg
          └── image2.jpg
```

## complete dataset structure
```
datasets/
├── Embeddings
├── RecycleBin
├── Trained_data
├── unTrained_data
      └── person_name1
          └── image1.jpg
          └── image2.jpg
      └── person_name2
          └── image1.jpg
          └── image2.jpg
```



