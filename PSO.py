import numpy as np
from multiprocessing import Pool
from utility import calculate_image_quality, apply_enhancement_parameters

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=2, c2=2):
        r1, r2 = np.random.rand(2, len(self.position))
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        for i, (low, high) in enumerate(bounds):
            self.position[i] = np.clip(self.position[i], low, high)
        self.position = np.clip(self.position, 0, 1)  # Ensure parameters are in [0, 1] range

    def evaluate(self, image):
        self.score = calculate_image_quality(image, self.position)
        if self.score < self.best_score:
            self.best_score = self.score
            self.best_position = self.position

def process_particle(particle_data):
    image, particle = particle_data
    particle.evaluate(image)
    return particle.best_score, particle.best_position

def pso(image, bounds, num_particles, num_iterations):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = np.zeros(len(bounds))
    global_best_score = float('inf')

    pool = Pool(processes=4)  # Adjust the number of processes based on your machine

    for _ in range(num_iterations):
        # Process each particle in parallel
        results = pool.map(process_particle, [(image, p) for p in particles])

        for score, position in results:
            if score < global_best_score:
                global_best_score = score
                global_best_position = position

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position(bounds)

    pool.close()
    pool.join()

    brightness_factor, contrast_factor = global_best_position
    return apply_enhancement_parameters(image, brightness_factor, contrast_factor)
