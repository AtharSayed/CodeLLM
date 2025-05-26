from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from functools import wraps
import os
import re

app = Flask(__name__)
app.secret_key = os.urandom(24)

# MongoDB configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['intellitube_db']
users_collection = db['users']

# Decorator to check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect('http://127.0.0.1:7860')
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form.get('email', '')  # Optional field in your template
        password = request.form['password']
        
        # Validate password strength
        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', password):
            flash('Password must be at least 8 characters with uppercase, lowercase, number, and special character', 'error')
            return redirect(url_for('signup'))
        
        # Check if user already exists
        if users_collection.find_one({'username': username}):
            flash('Username already taken!', 'error')
            return redirect(url_for('signup'))
        
        # Hash password and create user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        user_data = {
            'username': username,
            'email': email,
            'password': hashed_password
        }
        users_collection.insert_one(user_data)
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect('http://127.0.0.1:7860')
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html', error=request.args.get('error'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(port=5000, debug=True)