import cv2
import pytesseract
from PIL import Image
import numpy as np

# Set up the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Pandi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def recTextFromImage(image):
    # Convert the image to grayscale for better OCR results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # You can apply additional preprocessing like thresholding if needed
    # gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    
    # Convert the image to a format compatible with PIL
    pil_image = Image.fromarray(gray)
    
    # Use Tesseract to extract text
    text = pytesseract.image_to_string(pil_image)
    return text

# Open the live camera feed
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the live camera feed
    cv2.imshow("Live Camera Feed", frame)
    
    # OCR processing every few frames or on a specific trigger (e.g., pressing a key)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # Press 'r' to read the text from the current frame
        extracted_text = recTextFromImage(frame)
        print("Extracted Text:", extracted_text)
        
        # Optionally, write the extracted text to a file
        with open("LiveTextOutput.txt", "a") as file:
            file.write(extracted_text + "\n")
        print("Text written to LiveTextOutput.txt")
    
    # Exit loop when 'q' is pressed
    if key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
