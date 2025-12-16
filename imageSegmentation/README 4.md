# Image Segmentation Plugin for FLIKA

A powerful image segmentation plugin for FLIKA that implements multiple classification methods including traditional machine learning approaches and deep learning using U-Net architecture. This plugin is designed for analyzing and segmenting TIFF stacks with a focus on biological imaging applications.

## Features

- Multiple classification methods:
  - Traditional Machine Learning:
    - Random Forest
    - Support Vector Machines (SVM)
    - Gradient Boosting
    - k-Nearest Neighbors (k-NN)
  - Deep Learning:
    - U-Net architecture with customizable parameters

- Feature extraction options:
  - Original image
  - Gaussian blur
  - Sobel edge detection
  - Median filtering
  - Gradient magnitude

- Flexible U-Net implementation:
  - Support for both CPU and GPU processing
  - Customizable network architecture
  - Built-in data normalization
  - Training progress visualization
  - Model weights import/export

## Requirements

- FLIKA
- Python 3.x
- Required Python packages:
  ```
  numpy
  tensorflow
  scikit-learn
  scikit-image
  pytorch (optional - for PyTorch implementation)
  tqdm
  pandas
  matplotlib
  ```

## Installation

1. Ensure FLIKA is installed on your system
2. Clone this repository into your FLIKA plugins directory:
   ```bash
   cd ~/.FLIKA/plugins
   git clone https://github.com/yourusername/imageSegmentation.git
   ```
3. Install required dependencies:
   ```bash
   pip install numpy tensorflow scikit-learn scikit-image tqdm pandas matplotlib
   ```

## Usage

### Basic Usage

1. Launch FLIKA
2. Load your image stack using File > Open
3. Open the Image Segmentation plugin from the plugins menu
4. Select your desired classifier and parameters
5. Click "Run" to perform the segmentation

### Training a New Model

```python
from imageSegmenter import UNetClassifier

# Initialize the classifier
classifier = UNetClassifier(
    img_height=128,
    img_width=128,
    img_channels=1,
    num_test_images=10,
    use_gpu=True
)

# Load training data
classifier.load_data(
    input_dir='path/to/data',
    images_dir='path/to/images',
    masks_dir='path/to/masks'
)

# Build and train the model
classifier.build_model()
history = classifier.train_model(
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    save_weights_path='model_weights.h5'
)
```

### Using Pre-trained Models

```python
# Load pre-trained weights
classifier.load_pretrained_weights('path/to/weights.h5')

# Predict on new images
classifier.predict_images(
    input_folder='path/to/test/images',
    output_folder='path/to/output'
)
```

## GUI Features

The plugin provides a comprehensive GUI with the following features:

- **Main Options:**
  - Training window selection
  - Testing window selection
  - Mask window selection
  - Classifier selection
  - Export options for images and masks

- **Traditional ML Options:**
  - SVM kernel selection
  - Random Forest parameters (n_estimators, max_depth)
  - Gaussian sigma parameter
  - Feature selection checkboxes

- **U-Net Options:**
  - Training epochs control
  - Model weights management
  - Training history visualization

## Advanced Features

### Large Image Processing

The plugin includes support for processing large images using a tiling approach:

```python
classifier.predict_large_images(
    input_folder='path/to/large/images',
    output_folder='path/to/output',
    tile_size=128,
    overlap=16
)
```

### Custom Data Normalization

You can customize how your data is normalized:

```python
# Get global intensity range
min_intensity, max_intensity = classifier.get_intensity_range(input_folder)

# Normalize individual images
normalized_image = classifier.normalize_image(
    image,
    min_intensity,
    max_intensity
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

George Dickinson (george.dickinson@gmail.com)

## Acknowledgments

- FLIKA development team
- U-Net architecture paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
