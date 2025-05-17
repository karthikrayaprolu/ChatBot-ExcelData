from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Test the API connection
try:
    response = model.generate_content("Hello, are you working?")
    print("API Connection Successful!")
except Exception as e:
    print(f"API Error: {str(e)}")

app = Flask(__name__)
CORS(app)

# Store uploaded Excel data in memory
excel_data = {}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in ['xlsx', 'xls', 'csv']:
        return jsonify({"error": "Only .xlsx, .xls, and .csv files are allowed"}), 400
    
    try:
        if file_ext == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        excel_data['df'] = df
        return jsonify({
            "message": "File uploaded successfully",
            "columns": df.columns.tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    if 'df' not in excel_data:
        return jsonify({"response": "Please upload an Excel file first."})
    
    # Convert DataFrame to string representation
    df_info = excel_data['df'].to_string()
    
    # Create prompt for Gemini with clear formatting instructions
    prompt = f"""
Given this Excel data:
{df_info}

User question: {user_message}

Please analyze the data and answer the question.
Respond in clear, structured plain text. Do not use any special symbols, markdown, asterisks, hashtags, or formatting characters. Use only line breaks and indentation for structure. The response should be easy to read and well-organized.
"""

    try:
        # Get response from Gemini
        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
