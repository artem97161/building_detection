# Building Detection and Counting

This script processes images to detect buildings in satellite or aerial photos and count them.

It converts the image to grayscale, removes noise using median and bilateral filters, enhances local contrast with CLAHE, applies Otsu thresholding, performs morphological operations (opening and closing), finds contours, vectorizes them into polygons using `shapely`, and counts buildings above a minimum area. The processing steps are visualized using `matplotlib`.

## Installation

Make sure the required libraries are installed:

```bash
pip install opencv-python numpy shapely matplotlib
