# settings.py
from flask import flash, request, redirect, render_template, url_for
import json
import time

from auth import is_logged_in

SETTINGS_FILE = 'settings.json'

# Function to load settings from the JSON file
def load_settings():
    try:
        with open(SETTINGS_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {"model": "default-model", "tokens": 999, "temperature": 0.000}


# Function to save settings to the JSON file
def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as file:
        json.dump(settings, file, indent=4)


def settings_page():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))    

    if request.method == 'POST':
        # Retrieve updated values from the form
        model = request.form.get('model')
        tokens = int(request.form.get('tokens'))
        temperature = float(request.form.get('temperature'))
        store_p = request.form.get('store_p')

        # Save updated settings
        new_settings = {
            "model": model,
            "tokens": tokens,
            "temperature": temperature,
            "store_p": store_p
        }
        save_settings(new_settings)
        flash('Settings updated successfully!')
        time.sleep(2)
        return redirect(url_for('home'))

    # Load current settings and render the page
    current_settings = load_settings()
    return render_template('settings.html', settings=current_settings)

