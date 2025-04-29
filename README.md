# Cancer Detection Flask Application

![Application Interface](https://via.placeholder.com/400x300?text=Application+Interface)
![Results Page](https://via.placeholder.com/400x300?text=Results+Page)

# Technologies Used

- **Flask**: Web server framework
- **TensorFlow**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **OpenCV**: Image processing
- **Matplotlib**: Visualization tools
- **HTML/CSS**: Frontend interface

##  Installation 

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the repository

```bash
git clone https://github.com/baga6312/cancer-recognizer-tensorflow.git
cd cancer-recognizer-tensorflow 
```

### Step 2: Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
python app.py
```

##  API Usage

The application provides a RESTful API for programmatic access:

### Endpoint: `/api/predict`


##  Training Your Own Model

To train the model on your own dataset:

1. Prepare your dataset in the following structure:
```
dataset/
├── cancer/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── non_cancer/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

2. Run the training script:
```bash
python train_model.py --data_path dataset/ --epochs 20 --batch_size 32
```
