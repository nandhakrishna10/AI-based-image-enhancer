import numpy as np
import cv2

def calculate_image_quality(image, parameters):
    # Apply image enhancement parameters, parameters[0] is for brightness, parameters[1] for contrast
    enhanced_image = apply_enhancement_parameters(image, parameters[0], parameters[1])
    # Calculate and return the quality score of the image, for instance, using entropy
    return -1 * image_entropy(enhanced_image)

def apply_enhancement_parameters(image, brightness_factor, contrast_factor):
    # Apply adaptive histogram equalization using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    # Apply gamma correction for brightness adjustment
    gamma = 1.0 / brightness_factor
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, look_up_table)

    # Apply linear contrast scaling
    image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

    # Apply unsharp masking on the full-scale image to avoid losing details
    gaussian_blur = cv2.GaussianBlur(image, (9, 9), 0)
    image = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)

    return image



def image_entropy(image):
    """Calculate the entropy of an image."""
    histogram = cv2.calcHist([image],[0],None,[256],[0,256])
    histogram_length = sum(histogram)
    samples_probability = [float(h) / histogram_length for h in histogram]
    return -sum([p * np.log2(p) for p in samples_probability if p != 0])

# Add additional utility functions as necessary for your image processing
