import os
from flask import Flask, request, jsonify, render_template
from predict import load_trained_model, predict_image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max limit

# Global model variable to avoid reloading on every request
model = None

try:
    model = load_trained_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model could not be loaded. Please ensure 'models/cat_dog_classifier.onnx' exists. Error: {e}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 500

    # check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Perform prediction
            pred_class, confidence = predict_image(model, filepath)
            
            # Clean up the temp file
            os.remove(filepath)
            
            return jsonify({
                'class': pred_class,
                'confidence': f"{confidence:.2f}"
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg.'}), 400

# Vercel requires the app variable to be exposed, which we did.
# This block is for local development testing.
if __name__ == '__main__':
    app.run(debug=True, port=5000)
