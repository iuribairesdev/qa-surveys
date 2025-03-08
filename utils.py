
import os, json
from dotenv import load_dotenv
from flask import jsonify


# Allowed Extensions
ALLOWED_EXTENSIONS = {'csv'}

SETTINGS_FILE = 'settings.json'
# Define the folder to save uploaded files
UPLOAD_FOLDER = './uploaded_files'


def delete_file(file_path):
    print('FILE PATH, file_path')
    try:
        os.remove(file_path)
        print(f"{file_path} has been deleted successfully.")
    except FileNotFoundError:
        print(f"{file_path} does not exist.")
    except PermissionError:
        print(f"Permission denied: Cannot delete {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def delete_files(user):
    print('delete files ')
    try:
        # Delete PDF files (never store PDF files)
        files = os.listdir('./uploaded_files/')
        for f in files:
            os.remove('./uploaded_files/'+f)
    except FileNotFoundError:
        print(f"{f} does not exist.")
    except PermissionError:
        print(f"Permission denied: Cannot delete {f}.")
    except Exception as e:
        print(f"An error occurred: {e}")

    with open(SETTINGS_FILE, 'r') as file:
        settings = json.load(file)
    
    # if param store_p is no True
    if not settings['store_p']:    
        try:
            # Delete pickle objects
            files = os.listdir(os.path.join('./objects/', user, '/*'))
            print('files ', files)
            for f in files:
                os.remove(f)
        except FileNotFoundError:
            print(f"{f} does not exist.")
        except PermissionError:
            print(f"Permission denied: Cannot delete {f}.")
        except Exception as e:
            print(f"An error occurred: {e}")


# Create a helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to validate if the input is a valid list of strings
def validate_string(str1):
    if not isinstance(str1, str):
        return False, []
    # Convert comma-separated string to a list and strip any extra spaces
    str_list = [s.strip() for s in str1.split(',') if s.strip()]
    # If the list is empty, return False
    if len(str_list) == 0:
        return False, []
    return True, str_list



def save_csv(df, filename):
    # Step 6. Save to  csv file
    filename = filename + '-summary-' + datetime.datetime.now().strftime("%Y%m%d") + '.csv'
    filepath = os.path.join(
        UPLOAD_FOLDER
        ,filename)
    print('FILE', filepath)
    df.to_csv(filepath, index="False")

