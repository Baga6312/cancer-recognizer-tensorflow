import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "cancer_detection_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'cancer_detection_model.h5'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the model
model = None

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def build_model(input_shape=(128, 128, 3)):
    """Build and compile a CNN model for cancer detection"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (cancer or not)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess image for model prediction"""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def train_model(train_path=None):
    """
    Train the cancer detection model
    
    For a real application, you would include code here to load your training data
    and train the model with appropriate validation.
    
    This is a placeholder for demonstration purposes.
    """
    # In a real application, you would:
    # 1. Load and preprocess your cancer image dataset
    # 2. Split into training and validation sets
    # 3. Train the model
    # 4. Save the trained model
    
    # For demonstration, we'll just create a simple model and save it
    model = build_model()
    
    # Placeholder for training (in a real app, you'd train on actual data)
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    
    # Save the model
    model.save(app.config['MODEL_PATH'])
    return model

def load_trained_model():
    """Load the trained model or create a new one if it doesn't exist"""
    global model
    if os.path.exists(app.config['MODEL_PATH']):
        model = tf.keras.models.load_model(app.config['MODEL_PATH'])
        print("Loaded existing model")
    else:
        model = train_model()
        print("Created and saved new model")
    return model

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make prediction"""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image and make prediction
        processed_image = preprocess_image(filepath)
        prediction = model.predict(processed_image)[0][0]
        
        # Determine result based on prediction threshold
        is_cancer = prediction > 0.5
        confidence = prediction if is_cancer else 1 - prediction
        
        result = {
            'filename': filename,
            'is_cancer': bool(is_cancer),
            'confidence': float(confidence * 100),  # Convert to percentage
            'message': 'Cancer detected' if is_cancer else 'No cancer detected'
        }
        
        return render_template('result.html', result=result)
    
    flash('Invalid file type. Please upload an image file (png, jpg, jpeg).')
    return redirect(request.url)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image and make prediction
        processed_image = preprocess_image(filepath)
        prediction = model.predict(processed_image)[0][0]
        
        # Determine result based on prediction threshold
        is_cancer = prediction > 0.5
        confidence = prediction if is_cancer else 1 - prediction
        
        return jsonify({
            'filename': filename,
            'is_cancer': bool(is_cancer),
            'confidence': float(confidence * 100),  # Convert to percentage
            'message': 'Cancer detected' if is_cancer else 'No cancer detected'
        })
    
    return jsonify({'error': 'Invalid file type. Please upload an image file (png, jpg, jpeg).'}), 400

if __name__ == '__main__':
    # Load or create the model when the app starts
    load_trained_model()
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Cancer Detection App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .upload-form {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .upload-form input[type="file"] {
            margin: 10px 0;
        }
        .upload-form input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .upload-form input[type="submit"]:hover {
            background-color: #45a049;
        }
        .flash-messages {
            color: #f44336;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Cancer Detection App</h1>
    <p>Upload a medical image to check for signs of cancer.</p>
    
    <div class="flash-messages">
        {% for message in get_flashed_messages() %}
            <p>{{ message }}</p>
        {% endfor %}
    </div>
    
    <div class="upload-form">
        <h2>Upload Image</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".png, .jpg, .jpeg">
            <br>
            <input type="submit" value="Analyze Image">
        </form>
    </div>
    
    <div>
        <h2>About this application</h2>
        <p>This application uses a convolutional neural network trained on medical imaging data to detect potential cancerous tissue in uploaded images.</p>
        <p>Please note that this is a demonstration tool and should not be used for actual medical diagnosis. Always consult with a healthcare professional for medical advice.</p>
    </div>
</body>
</html>
        ''')
    
    # Create result.html
    with open('templates/result.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Cancer Detection Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .result-container {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .result-cancer {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }
        .result-no-cancer {
            background-color: #e8f5e9;
            border-left: 5px solid #4CAF50;
        }
        .confidence-meter {
            margin: 15px 0;
        }
        .confidence-bar {
            height: 20px;
            background-color: #ddd;
            border-radius: 10px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
        }
        .confidence-fill-cancer {
            background-color: #f44336;
        }
        .confidence-fill-no-cancer {
            background-color: #4CAF50;
        }
        .back-button {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 15px;
            background-color: #2196F3;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }
        .back-button:hover {
            background-color: #0b7dda;
        }
        .disclaimer {
            margin-top: 30px;
            padding: 10px;
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
        }
    </style>
</head>
<body>
    <h1>Analysis Result</h1>
    
    <div class="result-container {% if result.is_cancer %}result-cancer{% else %}result-no-cancer{% endif %}">
        <h2>{{ result.message }}</h2>
        <p>Filename: {{ result.filename }}</p>
        
        <div class="confidence-meter">
            <p>Confidence: {{ "%.2f"|format(result.confidence) }}%</p>
            <div class="confidence-bar">
                <div class="confidence-fill {% if result.is_cancer %}confidence-fill-cancer{% else %}confidence-fill-no-cancer{% endif %}" 
                     style="width: {{ result.confidence }}%"></div>
            </div>
        </div>
    </div>
    
    <a href="/" class="back-button">Upload Another Image</a>
    
    <div class="disclaimer">
        <h3>Important Disclaimer</h3>
        <p>This application is for demonstration purposes only and should not be used for actual medical diagnosis.</p>
        <p>The results provided are based on machine learning algorithms that may not be accurate for all cases.</p>
        <p>Always consult with a qualified healthcare professional for proper medical advice and diagnosis.</p>
    </div>
</body>
</html>
        ''')
    
    # Run the Flask app
    app.run(debug=True)
