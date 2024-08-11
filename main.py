import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)
    return edged

def detect_license_plate(edged, image):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        if len(approx) == 4:  # License plates are usually rectangular
            plate_contour = approx
            break
    
    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate = image[y:y+h, x:x+w]
        # Draw bounding box around the detected license plate in green
        cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)
        return plate, image
    else:
        return None, image

def enhance_plate_image(plate):
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    plate_blur = cv2.GaussianBlur(plate_gray, (5, 5), 0)
    plate_thresh = cv2.adaptiveThreshold(plate_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return plate_thresh

def extract_text_from_plate(plate):
    plate_thresh = enhance_plate_image(plate)
    config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(plate_thresh, config=config)
    return text.strip()

def recognize_license_plate(image):
    edged = preprocess_image(image)
    plate, image_with_box = detect_license_plate(edged, image)
    
    if plate is not None:
        text = extract_text_from_plate(plate)
        return text.strip(), image_with_box
    else:
        return "License plate not detected.", image_with_box

# Streamlit App
st.title("License Plate Recognition")
st.write("Upload an image, and the app will detect and display the license plate.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    result_text, processed_image = recognize_license_plate(image)
    
    st.image(processed_image, caption="Processed Image", use_column_width=True)
    st.write(f"Detected License Plate: {result_text}")
