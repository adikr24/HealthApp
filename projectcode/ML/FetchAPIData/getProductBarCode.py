import cv2
from pyzbar import pyzbar
import numpy as np

def preprocess_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Increase contrast and brightness
    alpha = 2.0  # Contrast control
    beta = 50    # Brightness control
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Apply adaptive thresholding to make bars stand out
    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    return img

def extract_barcode(image_path):
    img = preprocess_image(image_path)
    
    # Decode barcodes
    barcodes = pyzbar.decode(img)
    results = []
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        results.append({"type": barcode_type, "data": barcode_data})
    
    return results

if __name__ == "__main__":
    img_path = "/app/mediaFiles/images/ResnetTrainingImages/Protienbar/Screenshot 2025-08-22 193749.png"
    barcodes = extract_barcode(img_path)
    
    if not barcodes:
        print("No barcode detected.")
    else:
        for b in barcodes:
            print(f"Type: {b['type']}, Data: {b['data']}")
