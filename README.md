﻿# WIDS_2024

# YOLO Car Detection

This repository provides a Python script that detects cars in an image using the YOLOv8 model. Bounding boxes are drawn around detected cars, each labeled with an identifier.

## Requirements

Before using this script, ensure you have the following installed:

- Python 3.8+
- OpenCV
- ultralytics
- matplotlib

To install the required libraries, run:

```bash
pip install opencv-python ultralytics matplotlib
```

## Usage

1. Clone this repository or download the `main.py` file.

2. Place the image you want to process in the project directory.

3. Open the `main.py` file and update the `image_path` variable with the path to your image:

```python
image_path = 'your_image.jpg'  # Replace with the actual path to your car image
```

4. Run the script:

```bash
python main.py
```

5. The script will process the image, draw bounding boxes around detected cars, and save the output image in the same directory with the prefix `output_` added to the original filename. It will also display the processed image using matplotlib.





