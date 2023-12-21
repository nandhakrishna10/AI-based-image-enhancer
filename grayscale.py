import numpy as np
import cv2
import random
from skimage.metrics import structural_similarity as ssim

# Define the fitness function using SSIM and image variance
def fitness(image, original):
    ssim_value = ssim(image, original, data_range=image.max() - image.min())
    variance_value = np.var(image)
    return ssim_value + variance_value


# Define the genetic algorithm
# The genetic algorithm function
def genetic_algorithm(population_size, generations, mutation_rate, image):
    def contrast_adjust(image, alpha):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    population = [random.uniform(0.5, 2.0) for _ in range(population_size)]

    for generation in range(generations):
        scores = [fitness(contrast_adjust(image, alpha), image) for alpha in population]
        best_alpha = population[np.argmax(scores)]
        
        new_population = [best_alpha]  # Keep the best alpha
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            child = (parent1 + parent2) / 2  # Crossover
            if random.random() < mutation_rate:  # Mutation
                child += random.uniform(-0.2, 0.2)
            child = max(0.5, min(2.0, child))  # Ensure alpha is within the desired range
            new_population.append(child)

        population = new_population
    
    best_alpha = population[np.argmax([fitness(contrast_adjust(image, alpha), image) for alpha in population])]
    return contrast_adjust(image, best_alpha)

# Enhancement function to apply to the result of the genetic algorithm
def enhance_grayscale_image(image):
    # Apply CLAHE with adjusted parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_image = clahe.apply(image)
    
    # Apply a bilateral filter with adjusted parameters
    filtered_image = cv2.bilateralFilter(contrast_enhanced_image, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Apply unsharp masking with careful parameter selection
    gaussian_blur = cv2.GaussianBlur(filtered_image, (5, 5), 0)
    unsharp_image = cv2.addWeighted(filtered_image, 1.5, gaussian_blur, -0.5, 0)

    return unsharp_image


# Load the color image and convert it to grayscale
color_image = cv2.imread('images\color2.jpeg')
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Run the genetic algorithm to find a good contrast adjustment
ga_enhanced_image = genetic_algorithm(population_size=20, generations=100, mutation_rate=0.1, image=gray_image)

# Further enhance the image using the additional techniques
enhanced_image = enhance_grayscale_image(ga_enhanced_image)

# Save the final enhanced grayscale image
cv2.imwrite('results\enhanced_grayscale_image.png', enhanced_image)
