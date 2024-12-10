# README

## Face and Mask Detection

This project implements a face and mask detection system using deep learning techniques. The model is built using TensorFlow/Keras and includes functionality for data preprocessing, model training, and evaluation. Additionally, it employs OpenCV for image and video processing.

### Features
- Preprocesses images for training and validation.
- Builds a convolutional neural network (CNN) for mask detection.
- Evaluates the model's performance using metrics like accuracy and F1-score.
- Detects faces and identifies whether they are wearing masks in real-time.

### Requirements
To run this project, ensure you have the following installed:

- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- PIL (Pillow)

### Setup
1. Clone this repository.
2. Install dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the following pre-trained model files and place them in the specified directory:
   - `deploy.prototxt`: Defines the model architecture for face detection.
   - `res10_300x300_ssd_iter_140000.caffemodel`: Contains the pre-trained weights.
4. Download the dataset from the following link and place it in the dataset directory:
   - [Face Mask Detection Dataset](https://drive.google.com/drive/folders/1OeNy7zJLv0eGEm6T4L7G-dun6LC8Jx1D?usp=drive_link)
5. Run the notebook or scripts to train the model and perform detection.

### Usage
1. Prepare your dataset with labeled images of faces with and without masks.
2. Use the Jupyter Notebook to preprocess the data, train the model, and evaluate performance.
3. Run the detection module on images or video streams.

### Output
- Model accuracy and loss metrics.
- Real-time detection of faces and mask status.

---


