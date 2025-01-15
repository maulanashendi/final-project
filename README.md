# Flask Image Classification App

This project is part of my undergraduate thesis, focusing on utilizing machine learning for image classification.

This repository contains a Flask web application that performs image classification using a pre-trained Keras model. The application allows users to upload an image and get a prediction about the freshness of the image based on the model's inference.

## Features

- **Image Upload**: Users can upload images in `png`, `jpeg`, or `jpg` formats.
- **Image Classification**: The application uses a Keras model to classify the uploaded image into one of four categories: `segar`, `tidak segar`, `cukup segar`, or `sangat tidak segar`.
- **Dynamic Results**: The prediction result is displayed along with the uploaded image on the web interface.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.x
- Flask
- Keras
- TensorFlow
- NumPy

You can install the dependencies using pip:

```bash
pip install flask keras tensorflow numpy
```

## File Structure

- `app.py`: The main application script containing the Flask app.
- `templates/index.html`: The HTML template for the web interface.
- `static/`: Directory containing static files such as uploaded images and the pre-trained model (`model 1.0.h5`).

## How It Works

Convolutional Neural Networks (CNNs) are the foundation of this application. A CNN is a class of deep learning models specifically designed for image processing tasks. The core operations in a CNN include:

1. **Convolution**: Extracts features from the input image by applying a set of learnable filters (kernels). The convolution operation can be represented as:

   (I * K)(x, y) = Σ Σ I(x+m, y+n) * K(m, n)

   Where:
   - I is the input image.
   - K is the kernel/filter.
   - (x, y) are the coordinates of the pixel.
   - The summation computes the dot product of the kernel with the overlapping region of the image.

2. **Activation Function**: Introduces non-linearity into the network. A common choice is the ReLU (Rectified Linear Unit) function:

   f(x) = max(0, x)

3. **Pooling**: Reduces the spatial dimensions of the image while retaining important features. Common pooling methods include max pooling and average pooling.

4. **Flattening and Dense Layers**: After feature extraction, the resulting feature map is flattened into a vector and passed through fully connected (dense) layers to make the final prediction.

5. **Softmax Function**: Converts the final layer's output into probabilities for each class:

   σ(z)_i = e^(z_i) / Σ e^(z_j)

   Where z_i is the input to the softmax function for class i, and N is the total number of classes.

In this application, the uploaded image is processed and passed through a pre-trained CNN model. The model outputs a probability distribution over the predefined categories, which is interpreted and displayed as the final classification result.

1. **Home Page**: The user is presented with a form to upload an image.
2. **Image Validation**: The application validates the uploaded file to ensure it is in the allowed formats.
3. **Image Processing**: The image is preprocessed to match the input requirements of the model (resize to 224x224, convert to array, and normalize).
4. **Prediction**: The pre-trained Keras model makes a prediction on the processed image.
5. **Result Display**: The result is displayed on the same page along with the uploaded image.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-repo-url
```

2. Navigate to the project directory:

```bash
cd your-project-directory
```

3. Run the Flask application:

```bash
python app.py
```

4. Open your web browser and navigate to:

```
http://127.0.0.1:5000/
```

5. Upload an image and view the prediction results.

## Key Functions and Code Overview

### `allowed_file(filename)`

Checks if the uploaded file has a valid extension.

### Routes

- **`GET /`**: Renders the home page (`index.html`).
- **`POST /`**: Handles the image upload, processes the image, and returns the prediction result.

### Machine Learning Workflow

1. Load the pre-trained Keras model (`model 1.0.h5`).
2. Preprocess the image:
   - Resize to 224x224.
   - Convert to a NumPy array.
   - Normalize using `preprocess_input`.
3. Predict the class of the image.
4. Map the prediction to a human-readable label (`segar`, `tidak segar`, etc.).

## Error Handling

- Displays a flash message if the uploaded file is not valid.
- Handles cases where the model cannot classify the image.
