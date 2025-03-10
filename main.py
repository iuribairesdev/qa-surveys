import os, re, json
from flask import Flask, session, request, jsonify, redirect, send_file, render_template, url_for
from flask_session import Session
import pandas as pd

from dotenv import load_dotenv
from werkzeug.utils import secure_filename


from auth import is_logged_in, login, logout, init_oauth, auth_bp, google_login,  auth_callback
from prompts import get_prompt, edit_prompt, create_prompt, read_prompts, delete_prompt, prompts_page
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
        prompt = get_prompt(prompt_id)
       

        if prompt['title'] == 'Multiple Prompts':
            # print('prompts', prompts)
            columns = df.columns.tolist()
            # print('columns', columns)
            return render_template(
                'preview_multiple.html'
                ,filename=filename
                ,prompt_id=prompt_id
                ,columns=columns
                ,prompts=read_prompts()
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

# def post_to_openai(text, model="gpt-4o", tokens=3000, temperature=0.2) -> None:
def post_to_openai(text, pretext, posttext='', model="gpt-4o", tokens=3000, temperature=0.2):
    print('POST TO OPENAI')
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{pretext}"},
                {"role": "user", "content": f"{text} \n\n {posttext}"}
            ],
            max_tokens=tokens,  # how long the completion to be
            temperature=temperature, # creativity level
            # response_format={"type": "json_object"}
        )      
        # print('response', response)  
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
    

    
def categorization(input_file):
    print("categorization", input_file)
    df1 = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], input_file)}.csv")
    df2 = pd.DataFrame()
    for col in df1.columns:
        # Apply cleaning
        df2[col] = df1[col].apply(preprocess_text)
    save_csv(df2, input_file)  
    return df2.to_html(classes='table table-striped', index=False)





def summarization(input_file, prompt_id):
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
        for column in df.columns:
            print("COLUMN", column)
            combined_text = " ".join(str(item) for item in df[column].dropna() if isinstance(item, str))
            print('LEN COMB', len(combined_text.splitlines()))
            if (len(combined_text.splitlines())) > 0:
                prompt = get_prompt(prompt_id)
                summarized_data[column] = post_to_openai(combined_text, prompt['pretext'], prompt['posttext'])
        print("FINISH REQUESTS")                  
        # Step 5: Create a summary DataFrame
        df = pd.DataFrame([summarized_data])
        save_csv(df, input_file) 

        return df.to_html(classes='table table-striped', index=False)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400
   



def multiple_prompts(input_file, prompt_id, custom_prompt_ids, custom_prompts):
    print("Multiple Prompts")
    print('input_file', input_file)
    print('prompt_id', prompt_id)
    print('promptIDs', custom_prompt_ids)
    # print('custom_prompts', custom_prompts)

    df = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], input_file)}.csv")
    dfz = pd.DataFrame()  # output dataframe
    print('df COLUMNS', df.columns)
    # print(indexed_mapping)  # Debug
    for i in range(len(custom_prompt_ids)):
        print('custom ID', custom_prompt_ids[i])
        print('loopi', i)
        if custom_prompt_ids[i] == '0':
            summarized_data = {}
            print('Run custom', custom_prompt_ids[i])
            if custom_prompts[i]['column'] in df.columns:
                summarized_data[custom_prompts[i]['column']] = post_to_openai(df[custom_prompts[i]['column']], custom_prompts[i]['custom_value'])
            df1 = pd.DataFrame([summarized_data])
        else:

            prompt = get_prompt(custom_prompt_ids[i])
            if prompt['title'] == 'Summarization': 
                summarized_data = {}            
                combined_text = " ".join(str(item) for item in df[custom_prompts[i]['column']].dropna() if isinstance(item, str))
                if (len(combined_text.splitlines())) > 0:
                    print('Summarize text')
                    prompt = get_prompt(prompt_id)
                    summarized_data[custom_prompts[i]['column']] = post_to_openai(combined_text, prompt['pretext'])
                # Create a summary DataFrame row
                df1 = pd.DataFrame([summarized_data])
            elif prompt['title'] == 'Categorization':
                print('catgorize text')
                df1 = pd.DataFrame()
                # Apply cleaning - Create categorized rows
                df1[custom_prompts[i]['column']] = df[custom_prompts[i]['column']].apply(preprocess_text)
        dfz = pd.concat([dfz, df1], axis=1, ignore_index=True)
    dfz.columns = df.columns
    return dfz.to_html(classes='table table-striped', index=False)    



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
            prompt = get_prompt(prompt_id)
            if prompt['title'] == 'Categorization':
                result=categorization(filename)
            elif prompt['title'] == 'Summarization':
                result=summarization(filename, prompt_id)
            elif prompt['title'] == 'Multiple Prompts':
                custom_prompt_ids = request.form.getlist('custom_prompt_id') if 'custom_prompt_id' in request.form else []
                custom_prompts = {k: v for k, v in request.form.items() if k.startswith('custom_prompts')}
                # Convert to an array
                arr_prompts = [
                    {"column": re.search(r'\[(.*?)\]', key).group(1), "custom_value": value}
                    for key, value in custom_prompts.items()
                ]
                # print('arr', arr)       
                result= multiple_prompts(filename, prompt_id, custom_prompt_ids, arr_prompts)
                
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
