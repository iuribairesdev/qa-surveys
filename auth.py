from flask import request, redirect, render_template, session, url_for
import json
import os 
from dotenv import load_dotenv

load_dotenv()
# Hardcoded user credentials for demonstration purposes
users = json.loads(os.getenv('USERS'))
      
# Helper function to check if user is logged in
def is_logged_in():
    return 'user' in session

# Function for login page
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the credentials are valid
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('home'))
        else:
            error = "Invalid credentials. Please try again."
            return render_template('login.html', error=error)

    # Render the login form for GET requests
    return render_template('login.html')

# Function to log user out
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))