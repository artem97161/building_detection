import cv2
import numpy as np
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

image_path = "example.jpg"
bing = cv2.imread(image_path)
gray = cv2.cvtColor(bing, cv2.COLOR_BGR2GRAY)

# median filter - impulse noise removal
median_filtered = cv2.medianBlur(gray, 5)

# two-way filter — smoothing while preserving contours
bilateral_filtered = cv2.bilateralFilter(median_filtered, 19, 75, 75)

# Local contrast enhancement (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(bilateral_filtered)

_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# morphological processing
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# vectorization of contours into polygons
min_area = 600  # minimum area of the object
polygons = []

for cnt in contours:
    if cv2.contourArea(cnt) > min_area:
        cnt = cnt.squeeze()
        if len(cnt) >= 3:
            polygons.append(Polygon(cnt))

num_buildings = len(polygons)
print("Buildings found:", num_buildings)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original (Gray)")

plt.subplot(2, 3, 2)
plt.imshow(median_filtered, cmap='gray')
plt.title("Median filtering")

plt.subplot(2, 3, 3)
plt.imshow(bilateral_filtered, cmap='gray')
plt.title("Two-way filtration")

plt.subplot(2, 3, 4)
plt.imshow(enhanced, cmap='gray')
plt.title("Contrast enhancement (CLAHE)")

plt.subplot(2, 3, 5)
plt.imshow(binary, cmap='gray')
plt.title("Binarization and morphology")

plt.subplot(2, 3, 6)
output = bing.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"Discovered buildings: {num_buildings}")

plt.tight_layout()
plt.show()
