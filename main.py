import os
from flask import Flask, jsonify, request, redirect, send_file, render_template, url_for, session
from werkzeug.utils import secure_filename
import pandas as pd

import openai
import datetime
from dotenv import load_dotenv
from utils import allowed_file
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
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400        

        file = request.files['file']
        # If no file is selected
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Please, upload PDF files only!"}), 400

        # If file is valid and has allowed extension
        if file and allowed_file(file.filename):
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])   
            filename = secure_filename(file.filename).split(".")[0]
            file.save(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.csv")
            print('File successfully uploaded!')        
            # Read CSV content
            try:
                df = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.csv")
            except Exception as e:
                return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400
            # Convert dataframe to HTML table
            table_html = df.to_html(classes='table table-striped', index=False)
        return render_template(
            'preview.html'
            ,filename=filename
            ,content=table_html
        )
 


def post_to_openai(text, model="gpt-4o", tokens=3000, temperature=0.2) -> None:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a QA specialist that summarizes open answers from surveys."},
                {"role": "user", "content": f"The goal is to process the following answers in a consistent and detailed summary:\n{text}"}
            ],
            max_tokens=tokens,  # how long the completion to be
            temperature=temperature, # creativity level
            # response_format={"type": "json_object"}
        )        
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

# Route to display the file preview
@app.route('/result', methods=['GET', 'POST'])
def result():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))
    filename = ''
    filepath = ''
    if request.method == 'POST':
        if 'confirm' in request.form:
            load_dotenv()
            openai.api_key = os.environ.get("OPENAI_API_KEY1")
            try:
                filename = request.form['filename']
                df = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.csv")
                # Get open questions/answers
                # df1 = df.iloc[2:,[7,8,10,11,12,16,18,20,22,24,25,27,28,29,30,32]]

                # Normalize text columns to lowercase
                text_columns = [
                    "QA_Team_Composition", "Vendor_Names", "Manual_QAs_qty", "Automated_QAs_qty",
                    "Developers_qty", "Backend_tools", "Frontend_Automation", "Mobile_Automation",
                    "UnitTest_Automation", "Coverage_Testing_Tools", "Testing_Type", "Test_Management_Tools",
                    "QA_metrics", "QA_Challenges", "QA_Suggestions", "QA_AI_Tools"
                ]
                # df.columns = text_columns
                
                # Step 4: Summarize each column
                summarized_data = {}
                for column in df.columns:
                    print("COLUMN", column)
                    combined_text = " ".join(str(item) for item in df[column].dropna() if isinstance(item, str))
                    print('LEN COMB', len(combined_text.splitlines()))
                    if (len(combined_text.splitlines())) > 0:
                        summarized_data[column] = post_to_openai(combined_text)
                print("FINISH RESUESTS")                  
                # Step 5: Create a summary DataFrame
                summary_df = pd.DataFrame([summarized_data])
                # Step 6. Save to  csv file
                filepath = os.path.join(
                    app.config['UPLOAD_FOLDER']
                    ,filename + '-summary-' + datetime.datetime.now().strftime("%Y%m%d") + '.csv')
                
                print('result',summary_df)
                summary_df.to_csv(filepath, index="False")

                result = summary_df.to_html(classes='table table-striped', index=False)
            except Exception as e:
                return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400




        if 'cancel' in request.form:
            # Go back to the form
            return redirect(url_for('home'))
        elif 'download' in request.form:
            print("download")
            filepath = request.form['filepath']
            filename = request.form['filename']
            return send_file(
                filepath,
                as_attachment=True,  # Set to False if you want to view in the browser
                download_name=str(filename + '-summary.csv'),
                mimetype="text/csv"
            )

        else:
            result = 'No response from ChatGPT '
    return render_template('result.html', page_title="Summary Result", result=result, filename=filename, filepath=filepath)



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
