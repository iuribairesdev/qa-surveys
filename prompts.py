# prompt.py
import json
from flask import flash, request, redirect, render_template, url_for
from auth import is_logged_in
import os
PROMPT_FILE = './prompts.json'


# Helper function to read prompts from JSON file
def read_prompts():
    if not os.path.exists(PROMPT_FILE):
        return []
    with open(PROMPT_FILE, 'r') as file:
        return json.load(file)
    

def get_prompt(prompt_id):
    """Retrieve a prompt based on ID"""
    prompts = read_prompts()
    prompt = next((p for p in prompts if str(p['id']) == prompt_id), None)
    if not prompt:
        # return "Prompt not found."
        return jsonify({"error": "Invalid prompt_id"}), 400
    return prompt

# Helper function to write prompts to JSON file
def write_prompts(prompts):
    with open(PROMPT_FILE, 'w') as file:
        json.dump(prompts, file, indent=4)


# Function to create a new prompt
def create_prompt():
    if request.method == 'POST':
        title = request.form.get('title')
        pretext = request.form.get('pretext')
        posttext = request.form.get('posttext')
        print(title, pretext, posttext)
        if title and pretext and posttext:
            prompts = read_prompts()
            prompt_id = len(prompts) + 1
            new_prompt = {'id': prompt_id, 'title': title, 'pretext': pretext, 'posttext': posttext}
            prompts.append(new_prompt)
            write_prompts(prompts)
            flash('Prompt created successfully!')
            return redirect(url_for('prompts'))
        else:
            flash('Prompt text cannot be empty.')
    
    return render_template('prompt_form.html', action='Create', prompt=None)

# Function to edit an existing prompt
def edit_prompt(prompt_id):
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))
    
    prompts = read_prompts()
    prompt = next((p for p in prompts if p['id'] == prompt_id), None)
    
    if not prompt:
        flash('Prompt not found.')
        return redirect(url_for('prompts'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        pretext = request.form.get('pretext')
        posttext = request.form.get('posttext')
        table_p = request.form.get('table_p')
        print(title, pretext, posttext)
        if title and pretext and posttext:
            prompt['title'] = title
            prompt['pretext'] = pretext
            prompt['posttext'] = posttext
            prompt['table_p'] = table_p
            write_prompts(prompts)
            flash('Prompt updated successfully!')
            return redirect(url_for('prompts'))
        else:
            flash('Prompt text cannot be empty.')
    
    return render_template('prompt_form.html', action='Edit', prompt=prompt)

# Function to delete a prompt
def delete_prompt(prompt_id):
    prompts = read_prompts()
    prompt = next((p for p in prompts if p['id'] == prompt_id), None)
    
    if prompt:
        prompts = [p for p in prompts if p['id'] != prompt_id]
        write_prompts(prompts)
        flash('Prompt deleted successfully!')
    else:
        flash('Prompt not found.')
    
    return redirect(url_for('prompts'))





# Function for prompt page
def prompts_page():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('prompts.html', prompts=read_prompts())



