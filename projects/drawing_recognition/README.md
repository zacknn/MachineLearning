# Drawing Recognition Application

A machine learning-powered drawing recognition application built with Python. The app uses a trained neural network to recognize hand-drawn images in real-time through an interactive GUI.

## Features

- **Interactive Canvas**: Draw directly on a GUI canvas
- **Real-time Prediction**: Instant recognition of drawn objects
- **Multiple Categories**: Recognizes 7 different drawing types:
  - Cat
  - Car
  - Helicopter
  - Star
  - House
  - Cloud
  - Sun
- **User-Friendly Interface**: Simple buttons for prediction and clearing
- **CPU-Optimized**: Runs efficiently on CPU without requiring GPU

## Project Structure

```
drawing_recognition/
├── app/
│   ├── main.py              # Main application entry point
│   ├── canvas_widget.py     # Custom tkinter canvas widget for drawing
│   ├── predictor.py         # Model loading and prediction logic
│   └── __pycache__/         # Python cache files (ignored)
└── model/
    ├── model.ipynb          # Jupyter notebook for model training
    └── quickdraw_model.keras # Pre-trained Keras model
```

## Requirements

- Python 3.7+
- TensorFlow/Keras
- Pillow (PIL)
- tkinter (usually comes with Python)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd drawing_recognition
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install tensorflow pillow
```

## Usage

Run the application:

```bash
python app/main.py
```

1. A window will open with a white canvas
2. Draw an object using your mouse
3. Click the **"Predict"** button to get the AI's prediction
4. The prediction and confidence score will be displayed
5. Click **"Clear"** to erase the canvas and draw again

## How It Works

1. **Canvas Input**: The drawing is captured from the tkinter canvas
2. **Preprocessing**:
   - Resized to 28x28 pixels
   - Converted to grayscale
   - Normalized (pixel values divided by 255)
   - Color inverted to match training data format
3. **Model Prediction**: The preprocessed image is fed to the trained neural network
4. **Output**: The predicted category and confidence score are displayed

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 28x28 grayscale images
- **Output**: 7 classes (cat, car, helicopter, star, house, cloud, sun)
- **Training Data**: QuickDraw dataset

## Notes

- The app forces CPU usage to ensure compatibility across different systems
- TensorFlow logging is suppressed for cleaner console output
- For best results, draw simple, recognizable shapes

## Future Improvements

- Add more drawing categories
- Implement real-time confidence visualization
- Add the ability to save predictions and drawings
- Support for custom model training

## License

This project is part of a machine learning learning repository.

## Author

Created as part of machine learning experiments with TensorFlow and Keras.
