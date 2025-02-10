import os
from flask import Flask, jsonify, request, redirect, render_template, url_for, session
from werkzeug.utils import secure_filename

from utils import allowed_file, validate_string,  delete_files
from settings import settings_page
from auth import is_logged_in, login, logout
from prompts import edit_prompt, create_prompt, read_prompts, delete_prompt, prompts_page



# Initialize Flask application
app = Flask(__name__)




# Secret key to encrypt session data
app.secret_key = os.environ.get('SECRET_KEY')

# Define the folder to save uploaded files
UPLOAD_FOLDER = './uploaded_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


PROMPT_FILE = 'prompts.json'

# Route to display the file preview
@app.route('/preview', methods=['POST'])
def preview():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Get the list of files from the request
        files = request.files.getlist('file')
        
        # If no files were uploaded
        if not files or len(files) == 0:
            return jsonify({"error": "No file part"}), 400
        
        prompt_id = request.form.get('prompt_id')
         
        # Read the chosen prompt
        prompts = read_prompts()
        prompt = next((p for p in prompts if str(p['id']) == prompt_id), None)
        if not prompt:
            return jsonify({"error": "Invalid prompt_id"}), 400
        
        pretext = prompt['pretext']
        posttext = prompt['posttext'] + "Output must be organized in a table format, as in questions by answers."
 
        # Process each uploaded file
        for f in files:
            print('file', f)
            # Skip if filename is empty
            if f.filename == '':
                continue
            
            if not allowed_file(f.filename):
                return jsonify({"error": f"Invalid file type for {f.filename}. Please only upload CSV files!"}), 400
            
            # Secure and save the file
            folder_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cvs')
            # Make sure the folder exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
         
            filename_only = secure_filename(f.filename).rsplit('.', 1)[0]
            saved_path = os.path.join(folder_path, filename_only + '.cvs')
            f.save(saved_path)
            print(f'File {filename_only} successfully uploaded!')

        return render_template(
            'preview.html',
            
            filename=", ".join(saved_filenames),  # or pass them as a list
            prompt_id=prompt_id,
        )
 



# Route to display the file preview
@app.route('/result', methods=['GET', 'POST'])
def result():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))

    if request.method == 'POST':
        # get file name from form
        filename = request.form['filename']
        print("filename ", filename)
        
        #get the client name from form
        client_name = request.form['client_name']
        client_name_list = client_name.split(',')
        client_name = client_name.split(',')
        first_client_name = client_name_list[0] if isinstance(client_name_list, list) else client_name
        
        # Get prompt_id from form
        prompt_id = request.form['prompt_id']
        prompts = read_prompts()
        prompt = next((p for p in prompts if str(p['id']) == prompt_id), None)

        
      
        if 'confirm' in request.form:
            data = request.form.get('input_text')

          
      
        elif 'cancel' in request.form:
            # Go back to the form
            return redirect(url_for('home'))
        elif 'download' in request.form:
            summary = request.form['summary']
            result = summary
            return export_text(summary, "result_summary.docx", prompt['table_p'])
            
    else:
        result = 'No response from ChatGPT '
    # Apply pipe replacement
    result = replace_pipe_with_line_break(result)
    return render_template('result.html', page_title="Summary Result", result=result, filename=filename, prompt_id=prompt_id)


### 
# Prompts
###
# Route for create_prompt page
# @app.route('/create', methods=['GET', 'POST'])
app.add_url_rule('/create', 'create_prompt', create_prompt, methods=['GET', 'POST'])

# Route to delete a prompt
# @app.route('/delete/<int:prompt_id>')
app.add_url_rule('/delete/<int:prompt_id>', 'delete_prompt', delete_prompt)

# Route for prompt page
# @app.route('/prompts')
app.add_url_rule('/prompts', 'prompts', prompts_page)

# Route to edit an existing prompt
# @app.route('/edit/<int:prompt_id>', methods=['GET', 'POST'])
app.add_url_rule('/edit/<int:prompt_id>', 'edit_prompt', edit_prompt, methods=['GET', 'POST'])


### 
# Settings
###
# Route to display the settings page
# @app.route('/settings', methods=['GET', 'POST'])
# Route for the settings page
app.add_url_rule('/settings', 'settings', settings_page, methods=['GET', 'POST'])

# Route for the login page
app.add_url_rule('/login', 'login', login, methods=['GET', 'POST'])

# Route for logging out
app.add_url_rule('/logout', 'logout', logout)

# Route to handle the home page and file uploads
@app.route('/', methods=['GET', 'POST'])
def home():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))
 
    session.pop('_flashes', None)
    # GET request renders the upload form
    return render_template('home.html', prompts=read_prompts())



# Run the Flask app on localhost
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
    