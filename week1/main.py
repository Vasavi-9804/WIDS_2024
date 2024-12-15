import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('yolov8l.pt')  

def process_car_image(image_path):

    # Read the input image
    image = cv2.imread(image_path)

    # Perform inference using YOLO
    results = model(image_path)
    car_class_id = 2  # COCO class ID for 'car'

    # Filter results for the "car" class
    cars = [box for box in results[0].boxes if int(box.cls[0]) == car_class_id]

    # Draw bounding boxes and numbers
    for idx, car in enumerate(cars):
        x1, y1, x2, y2 = map(int, car.xyxy[0])  # Convert to integers
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Assign a number and display it
        cv2.putText(image, f"{idx + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the output image
    output_path = os.path.join(
        os.path.dirname(image_path), 
        f"output_{os.path.basename(image_path)}"
    )
    cv2.imwrite(output_path, image)

    # Optionally display the processed image using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Processed Image: {os.path.basename(image_path)}")
    plt.show()

    return output_path

# Example usage
image_path = '1015.jpg'  # Replace with the actual path to your car image
output_image_path = process_car_image(image_path)
print(f"Processed image saved at: {output_image_path}")
