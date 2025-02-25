import os
from flask import Flask, jsonify, request, redirect, send_file, render_template, url_for, session
from werkzeug.utils import secure_filename
import pandas as pd

import datetime
from dotenv import load_dotenv
from utils import allowed_file
from settings import settings_page
from auth import is_logged_in, login, logout
from prompts import edit_prompt, create_prompt, read_prompts, delete_prompt, prompts_page
from authlib.integrations.flask_client import OAuth

# Summarization
import openai


# Categorization
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# Initialize Flask application
app = Flask(__name__)

# Google OAuth Configuration
oauth = OAuth(app)
app.config['GOOGLE_CLIENT_ID'] = os.environ.get("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.environ.get("GOOGLE_CLIENT_SECRET")
app.config['GOOGLE_DISCOVERY_URL'] = "https://accounts.google.com/.well-known/openid-configuration"

google = oauth.register(
    name="google",
    client_id=app.config["GOOGLE_CLIENT_ID"],
    client_secret=app.config["GOOGLE_CLIENT_SECRET"],
    server_metadata_url=app.config["GOOGLE_DISCOVERY_URL"],
    client_kwargs={"scope": "openid email profile"},
)




# Secret key to encrypt session data
app.secret_key = os.environ.get('SECRET_KEY')

# Define the folder to save uploaded files
UPLOAD_FOLDER = './uploaded_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PROMPT_FILE = 'prompts.json'



## create mmutliple prompts funcionality as a stage before previw, so the user can select the prmopt and columns respectively


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
       
        return render_template(
            'preview.html'
            ,filename=filename
            ,prompt_id=prompt_id
            ,content=table_html
        )
 


# def post_to_openai(text, model="gpt-4o", tokens=3000, temperature=0.2) -> None:
def post_to_openai(text, model="gpt-4o", tokens=3000, temperature=0.2):
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


def save_csv(df, filename):
    # Step 6. Save to  csv file
    filename = filename + '-summary-' + datetime.datetime.now().strftime("%Y%m%d") + '.csv'
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER']
        ,filename)                
    df.to_csv(filepath, index="False")
    
def categorization(input_file):
    print("categorization", input_file)
    df1 = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], input_file)}.csv")
    df2 = pd.DataFrame()
    for col in df1.columns:
        # Apply cleaning
        df2[col] = df1[col].apply(preprocess_text)
    save_csv(df2, input_file)  
    return df2


def summarization(input_file):
    print(input_file)
    load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY1")
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
        for column in df.columns:
            print("COLUMN", column)
            combined_text = " ".join(str(item) for item in df[column].dropna() if isinstance(item, str))
            print('LEN COMB', len(combined_text.splitlines()))
            if (len(combined_text.splitlines())) > 0:
                summarized_data[column] = post_to_openai(combined_text)
        print("FINISH RESUESTS")                  
        # Step 5: Create a summary DataFrame
        summary_df = pd.DataFrame([summarized_data])
        save_csv(summary_df, input_file)
        print('TYPE', type(summary_df))
        return summary_df        
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400
   

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
                # result=summarization(filename)

                load_dotenv()
                openai.api_key = os.environ.get("OPENAI_API_KEY1")
                try:
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
                    result = pd.DataFrame([summarized_data])
                    save_csv(result, filename)
                    print('TYPE', type(result))
                    
                except Exception as e:
                    return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400




                print('summarization')
        if 'cancel' in request.form:
            # Go back to the form
            return redirect(url_for('home'))
        elif 'download' in request.form:
            return send_file(
                os.path.join(app.config['UPLOAD_FOLDER'], filename),
                as_attachment=True,  # Set to False if you want to view in the browser
                download_name=str(filename + '-summary.csv'),
                mimetype="text/csv"
            )
    else:
        result = 'Bad method request '
    print(type(result))
    result_html = result.to_html(classes='table table-striped', index=False)
    return render_template('result.html', page_title="Summary Result", result=result_html, filename=filename)



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


#app.add_url_rule('/google_login', 'google_login', methods=['GET', 'POST'])
@app.route('/google_login', methods=['GET', 'POST'])
def google_login():
    return google.authorize_redirect(url_for("auth_callback", _external=True))

#app.add_url_rule('/auth_callback', 'auth_callback', methods=['GET', 'POST'])
@app.route('/auth_callback', methods=['GET', 'POST'])
def auth_callback():
    token = google.authorize_access_token()
    user_info = token.get("userinfo")  # Get user info
    session["user"] = user_info  # Save user info in session
    return redirect(url_for("home"))


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
