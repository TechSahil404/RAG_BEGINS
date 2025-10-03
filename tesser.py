from pdf2image import convert_from_path
import pytesseract

# Replace with your scanned PDF path
pdf_path = "attention.pdf"

images = convert_from_path(pdf_path)
text = ""

for img in images:
    text += pytesseract.image_to_string(img)

print(text[:1000])  # First 1000 characters
