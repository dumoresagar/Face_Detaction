import dlib
import numpy as np
import cv2
import os
import sqlite3
import faiss
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# Load models
face_detector = dlib.cnn_face_detection_model_v1('face_recognition_models/models/mmod_human_face_detector.dat')
face_rec_model = dlib.face_recognition_model_v1('face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat')
landmark_predictor_68 = dlib.shape_predictor('face_recognition_models/models/shape_predictor_68_face_landmarks.dat')

# Initialize SQLite database for embeddings
def init_db():
    with sqlite3.connect('face_embeddings.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                gender TEXT,
                client_id TEXT,
                client_user_id TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                embedding BLOB,
                image_path TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognized_users (
                recognized_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT,
                gender TEXT,
                client_id TEXT,
                client_user_id TEXT,
                image_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        conn.commit()
        print("Initialized face_embeddings.db with users and embeddings tables.")

# Initialize databases
init_db()

# Initialize FAISS index
index = faiss.IndexFlatL2(128)  # Assuming 128-dimensional embeddings

# Function to get face embeddings
def get_face_embeddings(image):
    detected_faces = face_detector(image, 1)
    embeddings = []

    for face in detected_faces:
        shape = landmark_predictor_68(image, face.rect)
        face_embedding = face_rec_model.compute_face_descriptor(image, shape)
        embeddings.append(np.array(face_embedding))
    
    return embeddings

# Function to save user and embeddings
def save_user_and_embeddings(user_name, gender, client_id, client_user_id, image_path):
    with sqlite3.connect('face_embeddings.db') as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM users WHERE name = ?', (user_name,))
        user_row = cursor.fetchone()

        # If user doesn't exist, create a new entry
        if user_row is None:
            cursor.execute('INSERT INTO users (name, gender, client_id, client_user_id) VALUES (?, ?, ?, ?)', 
                           (user_name, gender, client_id, client_user_id))
            user_id = cursor.lastrowid
        else:
            user_id = user_row[0]  # Get existing user ID
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found or could not be loaded: {image_path}")
            return None  # Skip if image can't be loaded
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        embeddings = get_face_embeddings(image_rgb)

        ids = []
        
        if embeddings:
            for embedding in embeddings:
                embedding_blob = embedding.astype(np.float32)  # Ensure embedding is float32
                embedding_blob = embedding_blob.reshape(1, -1)  # Reshape for FAISS

                # Save to SQLite database
                cursor.execute('INSERT INTO embeddings (user_id, embedding, image_path) VALUES (?, ?, ?)', 
                               (user_id, embedding_blob.tobytes(), image_path))
                conn.commit()
                ids.append(cursor.lastrowid)  # Get the ID of the last inserted row

                # Add embedding to FAISS index
                index.add(embedding_blob)  # Add to FAISS index
    
    return ids

# Create directories for images if they don't exist
os.makedirs('test_folder', exist_ok=True)
os.makedirs('recognized_images', exist_ok=True)  # Directory for recognized images

# API endpoint to add a new image
@app.route('/add-image', methods=['POST'])
def add_image():
    user_name = request.form.get('name')
    gender = request.form.get('gender')
    client_id = request.form.get('client_id')
    client_user_id = request.form.get('client_user_id')
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file to the directory
    image_path = os.path.join('test_folder', f"{client_id}_{client_user_id}_{time.time()}.jpg")
    file.save(image_path)
    
    # Process the image and save embeddings
    saved_ids = save_user_and_embeddings(user_name, gender, client_id, client_user_id, image_path)
    if saved_ids:
        return jsonify({'message': 'Saved embeddings', 'ids': saved_ids}), 200
    
    return jsonify({'error': 'Failed to save embeddings'}), 500

# API endpoint to delete an image and its embeddings
@app.route('/delete-image', methods=['DELETE'])
def delete_image():
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'No user_id provided'}), 400
    
    try:
        with sqlite3.connect('face_embeddings.db') as conn:
            cursor = conn.cursor()
            # Fetch all embeddings for the user
            cursor.execute('SELECT id, image_path FROM embeddings WHERE user_id = ?', (user_id,))
            embeddings = cursor.fetchall()
            
            # Delete embeddings from the database
            cursor.execute('DELETE FROM embeddings WHERE user_id = ?', (user_id,))
            conn.commit()

            # Remove files from the filesystem
            for emb_id, image_path in embeddings:
                if os.path.exists(image_path):
                    os.remove(image_path)
            
            # Optionally, delete the user from the users table
            cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
            conn.commit()

        return jsonify({'message': 'Image and embeddings deleted successfully'}), 200
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Failed to delete image and embeddings'}), 500

# API endpoint to recognize faces from a new image
# @app.route('/recognize', methods=['POST'])
# def recognize():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Save the uploaded image directly into the recognized_images folder
#     image_path = os.path.join('recognized_images', f"{time.time()}.jpg")
#     file.save(image_path)

#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         return jsonify({'error': 'Could not load image'}), 400

#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     embeddings = get_face_embeddings(image_rgb)

#     results = []  # To store recognized user details

#     if not embeddings:
#         return jsonify({'message': 'No faces detected in the image.'}), 200

#     with sqlite3.connect('face_embeddings.db') as conn:
#         cursor = conn.cursor()
#         # Fetch all known embeddings from the database
#         cursor.execute('SELECT user_id, embedding FROM embeddings')
#         known_embeddings = {user_id: np.frombuffer(embedding, dtype=np.float32) for user_id, embedding in
#                             cursor.fetchall()}

#         for embedding in embeddings:
#             min_distance = float("inf")
#             recognized_user = None  # Variable to store recognized user info

#             # Compare the current embedding with known embeddings
#             for db_user_id, known_embedding in known_embeddings.items():
#                 distance = np.linalg.norm(embedding - known_embedding)
#                 if distance < min_distance:
#                     min_distance = distance
#                     recognized_user = db_user_id  # Store the user ID for recognition

#             # Check if the minimum distance is below the threshold
#             if min_distance < 0.6:  # Set your threshold for recognition
#                 # Get user details from the database
#                 cursor.execute('SELECT name, client_id FROM users WHERE user_id = ?', (recognized_user,))
#                 user_info = cursor.fetchone()

#                 if user_info:
#                     name, client_id = user_info
#                     results.append({
#                         'user_id': recognized_user,
#                         'name': name,
#                         'client_id': client_id,
#                         'distance': min_distance
#                     })
#                 else:
#                     print(f"No user info found for user_id: {recognized_user}")

#     if not results:
#         return jsonify({'message': 'No faces recognized.'}), 200

#     return jsonify({'results': results}), 200

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image directly into the recognized_images folder
    image_path = os.path.join('recognized_images', f"{time.time()}.jpg")
    file.save(image_path)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Could not load image'}), 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    embeddings = get_face_embeddings(image_rgb)

    if not embeddings:
        return jsonify({'message': 'No faces detected in the image.'}), 200

    results = []  # To store recognized results for each detected face

    # Connect to the database
    with sqlite3.connect('face_embeddings.db') as conn:
        cursor = conn.cursor()
        
        # Fetch all known embeddings from the database
        cursor.execute('SELECT user_id, embedding FROM embeddings')
        known_embeddings = {user_id: np.frombuffer(embedding, dtype=np.float32) for user_id, embedding in cursor.fetchall()}

        # For each detected face's embedding, find the best match in known embeddings
        for embedding in embeddings:
            recognized_faces = []  # To store all recognized faces within threshold

            for db_user_id, known_embedding in known_embeddings.items():
                distance = np.linalg.norm(embedding - known_embedding)

                # Check if distance is below the threshold for recognition
                if distance < 0.6:
                    # Get user details from the database
                    cursor.execute('SELECT name, client_id FROM users WHERE user_id = ?', (db_user_id,))
                    user_info = cursor.fetchone()

                    if user_info:
                        name, client_id = user_info
                        recognized_faces.append({
                            'user_id': db_user_id,
                            'name': name,
                            'client_id': client_id,
                            'distance': distance
                        })

            # If matches were found for the face, add them to the results
            if recognized_faces:
                results.append({
                    'face_id': len(results) + 1,  # Identifier for each face in the image
                    'matches': recognized_faces
                })

    # If no faces were recognized, return a message indicating so
    if not results:
        return jsonify({'message': 'No faces recognized.'}), 200

    return jsonify({'results': results}), 200

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5001)