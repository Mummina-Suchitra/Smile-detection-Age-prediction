from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename
from smile_age_predictor import predict_image


# ------------------- App Config -------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------- Load Models -------------------
smile_model = load_model("models/smile_model.h5")
age_model = load_model("models/age_model.h5")
age_classes = ["Child", "Teen", "Adult", "Senior"]  # Adjust based on your dataset

# ------------------- In-memory Users -------------------
users = {}  # For simplicity; replace with database in production

# ------------------- Helper Functions -------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Smile prediction
    smile_pred = smile_model.predict(img_array)
    smile_class = "Smiling 😊" if np.argmax(smile_pred) == 1 else "Not Smiling 😐"
    
    # Age prediction
    age_pred = age_model.predict(img_array)
    age_idx = np.argmax(age_pred)
    age_class = f"{age_classes[age_idx]} ({int(np.max(age_pred)*100)}%)" if age_idx < len(age_classes) else "Unknown"

    return smile_class, age_class

def save_base64_image(base64_str, filename):
    header, encoded = base64_str.split(",", 1)
    data = base64.b64decode(encoded)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, "wb") as f:
        f.write(data)
    return filepath

# ------------------- Routes -------------------

@app.route('/')
def home():
    if not session.get('username'):
        return redirect(url_for('login'))
    # Show latest capture if available
    latest_image = None
    images = sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True)
    if images:
        latest_image = url_for('uploaded_file', filename=images[0])
    return render_template('home.html', username=session['username'], latest_image=latest_image)

# ------------------- Signup/Login/Logout -------------------
@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return "Username already exists!"
        users[username] = password
        session['username'] = username
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) == password:
            session['username'] = username
            return redirect(url_for('home'))
        return "Invalid username or password!"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ------------------- Detection -------------------
@app.route('/index')
def index():
    if not session.get('username'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict_photo', methods=['POST'])
def predict_photo():
    if not session.get('username'):
        return jsonify({"error":"Unauthorized"}), 401
    data = request.get_json()
    img_base64 = data['imageBase64']
    filename = f"{session['username']}_{len(os.listdir(app.config['UPLOAD_FOLDER']))+1}.png"
    filepath = save_base64_image(img_base64, filename)
    
    smile, age = predict_image(filepath)
    
    return jsonify({"smile_result": smile, "age_result": age})

# ------------------- Gallery -------------------
@app.route('/gallery')
def gallery():
    if not session.get('username'):
        return redirect(url_for('login'))
    images = []
    for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # For simplicity, we re-run prediction (in production, save results to DB)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            smile, age = predict_image(img_path)
            images.append({'filename': filename, 'smile': smile, 'age': age})
    return render_template('gallery.html', images=images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ------------------- Run App -------------------
if __name__ == "__main__":
    app.run(debug=True)
