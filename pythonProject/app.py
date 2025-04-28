from flask import Flask, render_template, redirect, request, jsonify, session, url_for, flash
from flask_bcrypt import Bcrypt
from flask_socketio import SocketIO, emit
from flask_jwt_extended import JWTManager, create_access_token
import mysql.connector
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from email_validator import validate_email, EmailNotValidError
import joblib
import hashlib
from BERT import train_model, load_model, predict_message  # Import BERT functions
import os

password = os.getenv("MY_SECRET_PASSWORD")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['JWT_SECRET_KEY'] = 'jwtsecretkey'
bcrypt = Bcrypt(app)
socketio = SocketIO(app)
jwt = JWTManager(app)

# Load the model and vectorizer for BERT
model, vectorizer = load_model()

# If no model is loaded, train and save one
if model is None or vectorizer is None:
    train_model()
    model, vectorizer = load_model()

# Generate RSA keys (In a real app, securely store and manage these keys)
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=password,
        database="chat_app"
    )

# Home route that redirects to login
@app.route('/')
def home():
    return redirect('/login')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate email
        try:
            validate_email(email)
        except EmailNotValidError as e:
            flash(str(e), 'error')
            return render_template('login.html')

        # Get user from DB
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and bcrypt.check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['email'] = user['email']
            token = create_access_token(identity=user['email'])
            flash('Login successful!', 'success')
            return redirect('/chat')
        else:
            flash('Invalid credentials!', 'error')
            return render_template('login.html')

    return render_template('login.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Validate email
        try:
            validate_email(email)
        except EmailNotValidError as e:
            flash(str(e), 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Insert user into DB
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_password))
        conn.commit()
        cursor.close()
        conn.close()

        flash('Registration successful! Please log in.', 'success')
        return redirect('/login')

    return render_template('register.html')

# Chat route
@app.route('/chat')
def chat():
    if 'user_id' not in session:
        flash('Please login to access the chat.', 'error')
        return redirect('/login')
    return render_template('chat.html')

# Function to analyze if the message is offensive or not using the trained model
def is_message_offensive(message):
    return predict_message(message, model, vectorizer)

# Function to hash the message
def hash_message(message):
    return hashlib.sha256(message.encode()).hexdigest()

# Function to sign the message using RSA private key
def sign_message(message):
    message_hash = hash_message(message)
    signature = private_key.sign(
        message_hash.encode(),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    return message_hash, signature.hex()

# Socket.IO Events for real-time chat
@socketio.on('send_message')
def handle_send_message_event(data):
    message = data['message']
    user_email = session['email']

    # Check if the message is offensive using BERT model
    if is_message_offensive(message):
        # Log the blocked message in the database for review
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO flagged_messages (user_id, message) VALUES (%s, %s)", (session['user_id'], message))
        conn.commit()
        cursor.close()
        conn.close()

        # Notify the user that the message is blocked due to offensive content
        emit('message_blocked', {'message': 'Your message was blocked due to inappropriate content.'}, broadcast=True)
        return  # Stop further processing for this message

    # Encrypt the message and print to the terminal
    encrypted_message = encrypt_message(message)
    print(f"Encrypted message: {encrypted_message}")

    # Sign the message
    message_hash, signature = sign_message(message)

    # Log notarization in the database (for record-keeping)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO notarizations (message_hash, user_id, signature) VALUES (%s, %s, %s)",
        (message_hash, session['user_id'], signature)
    )
    conn.commit()
    cursor.close()
    conn.close()

    # Emit the original plaintext and encrypted message to all clients
    emit('display_message', {
        'email': user_email,
        'message': message,
        'encrypted_message': encrypted_message
    }, broadcast=True)

# Function to encrypt the message using RSA public key
def encrypt_message(message):
    encrypted_message = public_key.encrypt(
        message.encode(),
        padding.PKCS1v15()  # No hashing, just PKCS1v15 padding
    )
    return encrypted_message.hex()

# Function to decrypt the message using RSA private key
def decrypt_message(encrypted_message):
    decrypted_message = private_key.decrypt(
        bytes.fromhex(encrypted_message),
        padding.PKCS1v15()  # No hashing, just PKCS1v15 padding
    )
    return decrypted_message.decode()

# Utility route to list all routes for debugging
@app.route('/routes')
def show_routes():
    output = []
    for rule in app.url_map.iter_rules():
        output.append(str(rule))
    return jsonify(output)

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
