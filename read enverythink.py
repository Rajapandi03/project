import cv2
import pytesseract
from PIL import Image
import pyttsx3
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Pandi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

engine = pyttsx3.init()

def recTextFromImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)
    text = pytesseract.image_to_string(pil_image)
    return text

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("Live Camera Feed", frame)
    
    extracted_text = recTextFromImage(frame)
    
    if extracted_text.strip():
        print("Extracted Text:", extracted_text)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_with_text = frame.copy()
        y0, dy = 50, 30
        for i, line in enumerate(extracted_text.splitlines()):
            y = y0 + i * dy
            cv2.putText(frame_with_text, line, (50, y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Text Output", frame_with_text)
        
        engine.say(extracted_text)
        engine.runAndWait()
        
        with open("LiveTextOutput.txt", "a") as file:
            file.write(extracted_text + "\n")
        print("Text written to LiveTextOutput.txt")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
