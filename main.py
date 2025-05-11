import os, re
from flask import Flask, session, request, jsonify, redirect, send_file, render_template, url_for, g
from flask_session import Session
import pandas as pd
from werkzeug.utils import secure_filename

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


import secrets
import io

from auth import is_logged_in, login, logout, init_oauth, auth_bp, google_login,  auth_callback
from prompts import get_prompt, edit_prompt, create_prompt, read_prompts, delete_prompt, prompts_page
from settings import settings_page
from utils import allowed_file, save_file


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

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
GOOGLE_APP_ID = os.environ.get('GOOGLE_APP_ID')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
REDIRECT_URI = "http://localhost:8080/oauth2callback"

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def get_drive_service(access_token):
    creds = Credentials(token=access_token)
    service = build('drive',
                     'v3', credentials=creds)
    return service



def download_file_from_drive(service, file_id):
    # Get file metadata
    file_metadata = service.files().get(fileId=file_id, fields="name, mimeType").execute()
    file_name = file_metadata['name']
    mime_type = file_metadata['mimeType']

    # Prepare the correct request
    if mime_type == 'application/vnd.google-apps.spreadsheet':
        # Export as Excel
        request = service.files().export_media(
            fileId=file_id,
            mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        file_name += '.xlsx'
    elif mime_type.startswith('application/vnd.google-apps.'):
        raise ValueError(f"Unsupported Google Docs format for export: {mime_type}")
    else:
        # Regular binary file (like uploaded .xlsx)
        request = service.files().get_media(fileId=file_id)

    # Download the file
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

    # Save locally or return the file-like object
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_name}")   
    with open(file_path, 'wb') as f:
        f.write(fh.getvalue())
    
    return f"{file_name}"


# Route to display the file preview
@app.route('/preview', methods=['POST'])
def preview():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        file_id = request.form['google_drive_file_id']
        access_token = request.form['access_token']
        if file_id:
            if not file_id or not access_token:
                return jsonify({"error": "Missing Google Drive file ID or access token."}), 400                 
            service = get_drive_service(access_token)
            file_path = download_file_from_drive(service, file_id)
            filename = secure_filename(file_path).split(".")[0]
            filetype = secure_filename(file_path).split(".")[1]
                    
            print('FILE', file_path)
                

        else:


            # Check if the post request has the file part
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400        
            else:
                file = request.files['file']
                # If no file is selected
                if file.filename == '':
                    return jsonify({"error": "No selected file"}), 400

                if not allowed_file(file.filename):
                    return jsonify({"error": "Please, upload CSV or XLSX files only!"}), 400

                # If file is valid and has allowed extension
                print('FILE', file.filename)
                if file and allowed_file(file.filename):
                    if not os.path.exists(app.config['UPLOAD_FOLDER']):
                        os.makedirs(app.config['UPLOAD_FOLDER'])   

                    # df, filename = file_to_df(file)
                    filetype = secure_filename(file.filename).split(".")[1]
                    filename = secure_filename(file.filename).split(".")[0]
                    file.save(f"{os.path.join(UPLOAD_FOLDER, filename)}.{filetype}")
                    print('File successfully uploaded!')        


        # Read CSV content
        if filetype == 'csv':
            df = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.{filetype}").head(5)
        elif filetype == 'xlsx':
            df = pd.read_excel(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.{filetype}", engine='openpyxl').head(5)
            # df.columns = df.iloc[0]
            # df = df[1:].reset_index(drop=True)

        # Convert dataframe to HTML table
        table_html = df.to_html(classes='table table-striped', index=False)

        model = request.form.get('model') 
        
        # Read the chosen prompt
        prompt_id = request.form.get('prompt_id') 
        prompt = get_prompt(prompt_id)
    
        if prompt['title'] == 'Multiple Prompts':
            # print('prompts', prompts)
            columns = df.columns.tolist()
            # print('columns', columns)
            return render_template(
                'preview_multiple.html'
                ,filename=f"{filename}.{filetype}"
                ,model=model
                ,prompt=prompt
                ,columns=columns
                ,prompts=read_prompts()
                ,content=table_html
            )
        return render_template(
            'preview.html'
            ,filename=f"{filename}.{filetype}"
            ,model=model
            ,prompt=prompt
            ,content=table_html
        )
    return






















#####
## Automation Text Evaluation
#####

# def post_to_openai(text, model="gpt-4o", tokens=3000, temperature=0.2) -> None:
def post_to_openai(model, text, pretext, posttext='', tokens=3000, temperature=0.2):
    print('POST TO OPENAI')
    print('model', model)
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

        message_content = response["choices"][0]["message"]["content"]
        total_tokens = response["usage"]["total_tokens"]
        # input_tokens = response["usage"]["input_tokens"]
        # output_tokens = response["usage"]["output_tokens"]

        # Cost calculation based on OpenAI pricing
        COST_PER_1K_TOKENS = 0.03  # Example for GPT-4
        total_cost = (total_tokens / 1000) * COST_PER_1K_TOKENS
        
        return message_content, total_tokens, total_cost
        
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
    

    
def categorization(filename):
    print("categorization", filename)
    filetype = secure_filename(str(filename)).split(".")[1]
    filename = secure_filename(str(filename)).split(".")[0]
    if filetype == 'csv':
        df = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.{filetype}").head(5)
    elif filetype == 'xlsx':
        df = pd.read_excel(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.{filetype}", engine='openpyxl').head(5)
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
    dfz = pd.DataFrame()
    for col in df.columns:
        # Apply cleaning
        dfz[col] = df[col].apply(preprocess_text)
    save_file(dfz, f"categorization-{filename}.{filetype}")  
    return dfz


    




def summarization(filename, model, prompt_id):
    load_dotenv()
    print('SUMMARIZATION')
   
    filetype = secure_filename(filename).split(".")[1]
    filename = secure_filename(filename).split(".")[0]
    if filetype == 'csv':
        df = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.csv").head(5)
    elif filetype == 'xlsx':
        df = pd.read_excel(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.{filetype}", engine='openpyxl').head(5)
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    summarized_data = {}
    tokens = {}
    costs = {}
    for col in df.columns:
        print("COLUMN", col)
        combined_text = " ".join(str(item) for item in df[col].dropna() if isinstance(item, str))
        print('LEN COMB', len(combined_text.splitlines()))
        if (len(combined_text.splitlines())) > 0:
            prompt = get_prompt(prompt_id)
            summarized_data[col], tokens[col], costs[col] = post_to_openai(model, combined_text, prompt['pretext'], prompt['posttext'])
    print("FINISH REQUESTS")     

    df_data = pd.DataFrame([summarized_data])
    df_tokens = pd.DataFrame([tokens])
    df_costs = pd.DataFrame([costs])


    save_file(df_data, f"summarization-{filename}.{filetype}")
    save_file(df_tokens, f"tokens-{filename}.{filetype}") 
    save_file(df_costs, f"costs-{filename}.{filetype}")  
    
    return df_data, df_tokens, df_costs
    
   



def multiple_prompts(filename, model, prompt_id, custom_prompt_ids, custom_prompts):
    print("Multiple Prompts")
    print('input_file', filename)
    print('prompt_id', prompt_id)
    print('promptIDs', custom_prompt_ids)
    # print('custom_prompts', custom_prompts)


    code_to_name = {
        '1': 'Summarization',
        '2': 'Categorization',
        '0': 'Custom',
        '-1': 'None'
    }
    # Convert codes to names
    labels = [code_to_name[code] for code in custom_prompt_ids]
    # Create the DataFrame with a single row
    df_labels = pd.DataFrame([labels])

    filetype = secure_filename(filename).split(".")[1]
    filename = secure_filename(filename).split(".")[0]
    if filetype == 'csv':
        df = pd.read_csv(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.{filetype}").head(5)
    elif filetype == 'xlsx':
        df = pd.read_excel(f"{os.path.join(app.config['UPLOAD_FOLDER'], filename)}.{filetype}", engine='openpyxl').head(5)
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)


    columns = df.columns
    # print(len(columns), columns)
    
    dfz_data = pd.DataFrame()
    dfz_tokens = pd.DataFrame()
    dfz_costs = pd.DataFrame()  # output dataframe
    
    # Replace NaN column names with a placeholder like "Unnamed_{index}"
    df.columns = [f"Unnamed_{i}" if pd.isna(col) else col for i, col in enumerate(df.columns)]
    for i, prompt in enumerate(custom_prompts):
        if prompt['column'] == 'nan':
            prompt['column'] = f'Unnamed_{i}'

    # print(df.columns, custom_prompts)
    

    for i in range(len(custom_prompt_ids)):
        print('custom ID', custom_prompt_ids[i])
        print('loopi', i)
        data = {}
        tokens = {}
        costs = {}
        

        if custom_prompt_ids[i] == '-1': # None: no processing
            data[custom_prompts[i]['column']] = df[custom_prompts[i]['column']]
            tokens[custom_prompts[i]['column']] = 0
            costs[custom_prompts[i]['column']] = 0

        elif custom_prompt_ids[i] == '0': # custom prompt
            print('Run custom', custom_prompt_ids[i])
            if custom_prompts[i]['column'] in df.columns:
                data[custom_prompts[i]['column']], tokens[custom_prompts[i]['column']], costs[custom_prompts[i]['column']] = post_to_openai(model, df[custom_prompts[i]['column']], custom_prompts[i]['custom_value'])
        else:
            
            prompt = get_prompt(custom_prompt_ids[i])
            print('title', prompt['title'])
            if prompt['title'] == 'Summarization':                
                combined_text = " ".join(str(item) for item in df[custom_prompts[i]['column']].dropna())
                if (len(combined_text.splitlines())) > 0:                    
                    data[custom_prompts[i]['column']], tokens[custom_prompts[i]['column']], costs[custom_prompts[i]['column']] = post_to_openai(model, combined_text, prompt['pretext'])
                
            elif prompt['title'] == 'Categorization':
                # Apply cleaning - Create categorized rows
                data[custom_prompts[i]['column']] = df[custom_prompts[i]['column']].apply(preprocess_text)
                tokens[custom_prompts[i]['column']] = 0
                costs[custom_prompts[i]['column']] = 0

        dfz_data = pd.concat([dfz_data, pd.DataFrame([data])], axis=1, ignore_index=True)
        dfz_tokens = pd.concat([dfz_tokens, pd.DataFrame([tokens])], axis=1, ignore_index=True)
        dfz_costs = pd.concat([dfz_costs, pd.DataFrame([costs])], axis=1, ignore_index=True)
    
    
    dfz = pd.concat([df_labels, dfz_data], ignore_index=True) 
    dfz.columns = columns
    return dfz, dfz_tokens, dfz_costs        



# Route to display the file preview
@app.route('/result', methods=['GET', 'POST'])
def result():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))
    filename = ''

    if request.method == 'POST':
        if 'cancel' in request.form:
            # Go back to the form
            return redirect(url_for('home'))
        filename = request.form['filename']
    
        if filename != '':    
            if 'confirm' in request.form:
                # Read chosen model
                model = request.form['model']
                # Read the chosen prompt
                prompt_id = request.form['prompt_id']
                print("prmpt_id", prompt_id)
                prompt = get_prompt(prompt_id)
                print("prompt", prompt)

                if prompt['title'] == 'Categorization':
                    data = categorization(filename)
                    totals_html = ''
                elif prompt['title'] == 'Summarization':
                    data, tokens, costs = summarization(filename, model, prompt_id)
                    df_totals = pd.DataFrame([{
                        "Output Tokens": tokens.loc[:,:].sum(axis=1)[0],
                        "cost $USD": costs.loc[:,:].sum(axis=1)[0]
                    }])
                    totals_html = df_totals.to_html(classes='table table-striped', index=False)
        
                elif prompt['title'] == 'Multiple Prompts':
                    custom_prompt_ids = request.form.getlist('custom_prompt_id') if 'custom_prompt_id' in request.form else []
                    custom_prompts = {k: v for k, v in request.form.items() if k.startswith('custom_prompts')}
                    # Convert to an array
                    arr_prompts = [
                        {"column": re.search(r'\[(.*?)\]', key).group(1), "custom_value": value}
                        for key, value in custom_prompts.items()
                    ]
                    # print('arr', arr)       
                    data, tokens, costs = multiple_prompts(filename, model, prompt_id, custom_prompt_ids, arr_prompts)
                    df_totals = pd.DataFrame([{
                        "Output Tokens": tokens.loc[:,:].sum(axis=1)[0],
                        "cost $USD": costs.loc[:,:].sum(axis=1)[0]
                    }])
                    totals_html = df_totals.to_html(classes='table table-striped', index=False)
                    
        
            elif 'download' in request.form:
                filetype = secure_filename(filename).split(".")[1]
                filename = secure_filename(filename).split(".")[0]
                type = str(request.form['type']).lower()
                filename = f"{type}-{filename}-{datetime.datetime.now().strftime("%Y%m%d")}.{filetype}"
                return send_file(
                    os.path.join(app.config['UPLOAD_FOLDER'], filename),
                    as_attachment=True,  # Set to False if you want to view in the browser
                    download_name=str(filename),
                    mimetype="application/xlsx"
                )
    else:
        data = 'Bad method request '

    data_html = data.to_html(classes='table table-striped', index=False)     

    return render_template('result.html', page_title=prompt['title'], result_html=data_html, totals_html=totals_html, filename=filename)



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


@app.before_request
def generate_nonce():
    g.nonce = secrets.token_urlsafe(16)
    
@app.after_request
def add_csp_headers(response):
    csp = (
        "default-src 'self'; ",
        f"script-src 'self' '{g.nonce}' https://apis.google.com https://www.gstatic.com "
        "https://accounts.google.com https://code.jquery.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "connect-src 'self' https://www.googleapis.com https://oauth2.googleapis.com; "
        "img-src 'self' data: https://ssl.gstatic.com https://www.gstatic.com; "
        "frame-src https://accounts.google.com https://content.googleapis.com "
        "https://docs.google.com https://drive.google.com; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'self';"
    )
    response.headers['Content-Security-Policy'] = app
    return response




# Route to handle the home page and file uploads
@app.route('/', methods=['GET', 'POST'])
def home():
    # If user is not logged in, redirect to login page
    if not is_logged_in():
        return redirect(url_for('login'))
 
    session.pop('_flashes', None)
    # GET request renders the upload form
  
    # GET request renders the upload form
    return render_template('home.html', nonce=g.nonce, GOOGLE_API_KEY=GOOGLE_API_KEY, GOOGLE_CLIENT_ID=GOOGLE_CLIENT_ID, GOOGLE_APP_ID=GOOGLE_APP_ID, prompts=read_prompts())



# Run the Flask app on localhost
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
