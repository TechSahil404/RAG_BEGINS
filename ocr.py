from PIL import Image
import pytesseract

# Agar PATH add nahi kiya installer me, to explicitly set karo:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Test ke liye ek simple image (replace with apni image)
img_path = "example2.png"  # ye image me kuch text hona chahiye
img = Image.open(img_path)

# OCR run karo
text = pytesseract.image_to_string(img)

print("Detected text from image:")
print(text)
