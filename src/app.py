import os
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import time
from extract_single_glb_embeddings import convert_glb_to_ply, load_ply, init_openshape_model
import joblib

def init_model():
    print("model initialized")
    return "model"

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'glb', 'GLB'}

# Create upload directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize model (placeholder)
#model = init_model()
model, config = init_openshape_model()
print("openshape model initialized")
#print(os.getcwd())

# init pca model
pca_model = joblib.load(os.path.join(os.getcwd(), "src/openshape_pca_1.2_to_1.6.joblib"))

# Feature extraction (placeholder)
def extract_feature(model, ply_path):
    xyz, feat = load_ply(ply_path)
    
    embedding = model(xyz, feat, device='cuda', quantization_size=config.model.voxel_size) 
    
    return embedding
    
    # time.sleep(1)
    # return str([0.12] * 1280) #"[1.23, 4.56, 7.89, 0.12, 3.45]"
# Check allowed files
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Initialize model before first request
# def initialize_model():
#     global model
#     model = init_model()
    
# @app.before_request
# def before_request_handler():
#     initialize_model()
    
# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Secure filename and add timestamp prefix
        filename = secure_filename(file.filename)
        unique_filename = f"{datetime.now().timestamp()}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(save_path)
        
        # Extract features
        ply_path = save_path + ".ply"
        convert_glb_to_ply(save_path, ply_path)
        
        openshape_embedding = extract_feature(model, ply_path).detach().cpu().numpy().squeeze()

        # optional: go through pca to search on data website
        openshape_embedding = openshape_embedding / np.max(np.abs(openshape_embedding))
        openshape_embedding = np.clip(openshape_embedding * 128, -128, 127) / 128.0

        openshape_embedding = pca_model.transform((openshape_embedding - np.mean(openshape_embedding))[None]).squeeze()

        full_vector_str = "[" + ', '.join([f"{x:.7f}" for x in openshape_embedding]) + "]"

        display_vector_str = np.array2string(openshape_embedding, edgeitems=3, separator=', ', formatter={'float_kind':lambda x: "%.7f" % x}, max_line_width=np.inf)

        # remove the file
        os.remove(save_path)
        os.remove(ply_path)
        
        # Create URL for the uploaded image
        image_url = url_for('uploaded_file', filename=unique_filename)
        
        return render_template('result.html', display_vector=display_vector_str, full_vector=full_vector_str, image_url=image_url)
    
    return 'Invalid file format. Please upload a GLB image.'

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
    #app.run(debug=True)
