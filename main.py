from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
import os
from PSO import pso  # Make sure this is correctly imported from your PSO script
from grayscale import *  # Make sure this is correctly imported from your grayscale script

app = Flask(__name__, static_folder='static')

# Ensure the 'temp' directory exists
temp_directory = 'temp'
os.makedirs(temp_directory, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return "No image part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    enhancement_type = request.form['enhancement_type']
    
    filename = secure_filename(file.filename)
    input_path = os.path.join('temp', filename)
    file.save(input_path)

    # Load the image for processing
    image = cv2.imread(input_path)
       
    if enhancement_type == 'low_light':
        # Define bounds for brightness and contrast parameters
        parameter_bounds = [(0.5, 1.2), (0.5, 1.2)]  # Adjust these values as needed

        # Run PSO to enhance the low-light image
        enhanced_image = pso(image, parameter_bounds, num_particles=30, num_iterations=10)
        
    elif enhancement_type == 'grayscale':
        # Convert to grayscale and enhance
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run the genetic algorithm to find a good contrast adjustment
        ga_enhanced_image = genetic_algorithm(population_size=20, generations=100, mutation_rate=0.1, image=gray_image)

        # Further enhance the image using the additional techniques
        enhanced_image = enhance_grayscale_image(ga_enhanced_image)
    else:
        return "Invalid enhancement type selected.", 400

    temp_output_path = os.path.join('temp', 'enhanced_image.png')
    cv2.imwrite(temp_output_path, enhanced_image)
    
    return send_file(temp_output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
