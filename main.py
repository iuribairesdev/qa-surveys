import os, re, json
from flask import Flask, session, request, jsonify, redirect, send_file, render_template, url_for
from flask_session import Session
import pandas as pd

from dotenv import load_dotenv
from werkzeug.utils import secure_filename


from auth import is_logged_in, login, logout, init_oauth, auth_bp, google_login,  auth_callback
from prompts import edit_prompt, create_prompt, read_prompts, delete_prompt, prompts_page
from settings import settings_page
from utils import allowed_file, save_csv


# Summarization
import openai
import pandas as pd
import datetime
# Categorization
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize Flask application
app = Flask(__name__)
# Secret key to encrypt session data
app.secret_key = os.environ.get('SECRET_KEY')

# Flask Session
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

init_oauth(app)
app.register_blueprint(auth_bp)


# Define the folder to save uploaded files
UPLOAD_FOLDER = './uploaded_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PROMPT_FILE = 'prompts.json'



# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')



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
            return jsonify({"error": "Please, upload CSV files only!"}), 400

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

        # Read the chosen prompt
        prompt_id = request.form.get('prompt_id') 
        prompts = read_prompts()
        prompt = next((p for p in prompts if str(p['id']) == prompt_id), None)
        if not prompt:
            return jsonify({"error": "Invalid prompt_id"}), 400

        if prompt['title'] == 'Multiple Prompts':
            # print('prompts', prompts)
            columns = df.columns.tolist()
            # print('columns', columns)
            return render_template(
                'preview_multiple.html'
                ,filename=filename
                ,prompt_id=prompt_id
                ,columns=columns
                ,prompts=prompts
                ,content=table_html
            )
        return render_template(
            'preview.html'
            ,filename=filename
            ,prompt_id=prompt_id
            ,content=table_html
        )
 






















#####
## Automation Text Evaluation
#####


def multiple_prompts(input_file, prompt_ids, custom_prompts, columns):
    print("Multiple Prompts")
    print(input_file, prompt_ids, custom_prompts, columns)

    # Clean the keys with regex and restructure
    custom_columns = {
        re.search(r'\[(.*?)\]', k).group(1): v for k, v in custom_prompts.items() if v.strip()
    }

    # Associate indexes with custom_mapping
    indexed_mapping = []
    for i, column in enumerate(columns):
        custom_value = custom_prompts.get(column, "")
        indexed_mapping.append({
            "index": i,
            "column": column,
            "custom_value": custom_value
        })

    print(indexed_mapping)  # Debug
    for i in range(len(prompt_ids)):
        if prompt_ids[i] == '0':
            print('Run custom',indexed_mapping[i])

        
    
    
# def post_to_openai(text, model="gpt-4o", tokens=3000, temperature=0.2) -> None:
def post_to_openai(text, model="gpt-4o", tokens=3000, temperature=0.2):
    print('POST TO OPENAI')
    openai.api_key = os.environ.get("OPENAI_API_KEY")
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
        print('response', response)  
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"


def summarization(input_file):
    load_dotenv()
    print('SUMMARIZATION')
    print('INPUT FILE',input_file)
    try:
        df = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], input_file)}.csv")
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
        print()
        for column in df.columns:
            print("COLUMN", column)
            combined_text = " ".join(str(item) for item in df[column].dropna() if isinstance(item, str))
            print('LEN COMB', len(combined_text.splitlines()))
            if (len(combined_text.splitlines())) > 0:
                summarized_data[column] = post_to_openai(combined_text)
        print("FINISH REQUESTS")                  
        # Step 5: Create a summary DataFrame
        df = pd.DataFrame([summarized_data])
        save_csv(df, input_file) 

        return df.to_html(classes='table table-striped', index=False)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400
   




# Preprocessing Function
def preprocess_text(text):
    # Handle numeric responses
    if isinstance(text, (int, float)):
        return str(text)  # Convert numeric response to string

    if not isinstance(text, str):
        return ""  # Return empty string for invalid types

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)
    

    
def categorization(input_file):
    print("categorization", input_file)
    df1 = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], input_file)}.csv")
    df2 = pd.DataFrame()
    for col in df1.columns:
        # Apply cleaning
        df2[col] = df1[col].apply(preprocess_text)
    save_csv(df2, input_file)  
    return df2.to_html(classes='table table-striped', index=False)



# Route to display the file preview
@app.route('/result', methods=['GET', 'POST'])
def result():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))
    filename = ''

    if request.method == 'POST':
        filename = request.form['filename']
        if 'confirm' in request.form:
            # Read the chosen prompt
            prompt_id = request.form['prompt_id']
            prompts = read_prompts()
            prompt = next((p for p in prompts if str(p['id']) == prompt_id), None)
            if not prompt:
                return jsonify({"error": "Invalid prompt_id"}), 400
            if prompt['title'] == 'Categorization':
                result=categorization(filename)
            elif prompt['title'] == 'Summarization':
                result=summarization(filename)
            elif prompt['title'] == 'Multiple Prompts':
                prompt_ids = request.form.getlist('custom_prompt_id') if 'custom_prompt_id' in request.form else []
                columns = json.loads(request.form.get('columns').replace("'", '"'))
                custom_prompts = {k: v for k, v in request.form.items() if k.startswith('custom_prompts')}
                       
                result=multiple_prompts(filename, prompt_ids, custom_prompts, columns)
                
        if 'cancel' in request.form:
            # Go back to the form
            return redirect(url_for('home'))
        elif 'download' in request.form:
            filename = filename + '-summary-' + datetime.datetime.now().strftime("%Y%m%d") + '.csv'
            return send_file(
                os.path.join(app.config['UPLOAD_FOLDER'], filename),
                as_attachment=True,  # Set to False if you want to view in the browser
                download_name=str(filename),
                mimetype="text/csv"
            )
    else:
        result = 'Bad method request '
    print('result', result)
    return render_template('result.html', page_title="Summary Result", result_html=result, filename=filename)



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

app.add_url_rule('/google_login', 'google_login', google_login, methods=['GET', 'POST'])

app.add_url_rule('/auth_callback', 'auth_callback', auth_callback, methods=['GET', 'POST'])

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
  
    # GET request renders the upload form
    return render_template('home.html', prompts=read_prompts())



# Run the Flask app on localhost
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
