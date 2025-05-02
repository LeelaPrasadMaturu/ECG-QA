from collections import Counter
from PIL import Image

# Load the sample ECG image
img = Image.open("EcgImages/ECG00021_clinical_512.png")

# Get pixel data
pixels = list(img.getdata())

# Count pixel values
pixel_distribution = Counter(pixels)

# Print the pixel distribution
print(pixel_distribution)  # Should show mostly 0 (black) and some 255 (white)