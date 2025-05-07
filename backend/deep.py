from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure OpenAI client with OpenRouter base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY')
)

# Extra headers for OpenRouter
EXTRA_HEADERS = {
    "HTTP-Referer": os.getenv('SITE_URL'),
    "X-Title": os.getenv('SITE_NAME')
}

# Test the API connection
try:
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[{"role": "user", "content": "Hello, are you working?"}],
        extra_headers=EXTRA_HEADERS
    )
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
        
    if not file.filename.endswith('.xlsx'):
        return jsonify({"error": "Only .xlsx files are allowed"}), 400
    
    try:
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
    
    try:
        # Get response from OpenRouter/Deepseek
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[
                {"role": "system", "content": """You are a helpful assistant that analyzes Excel data.
                Format your responses in a clear, readable manner without unnecessary symbols.
                Use proper spacing and clear headings.
                Avoid using '#' or '###' for headings.
                Use proper paragraphs and bullet points where appropriate."""},
                {"role": "user", "content": f"""Given this Excel data:
{df_info}

User question: {user_message}

Please analyze the data and provide a clear, well-formatted response."""}
            ],
            extra_headers=EXTRA_HEADERS
        )
        
        # Clean up the response
        cleaned_response = response.choices[0].message.content.replace('###', '').replace('#', '').strip()
        return jsonify({"response": cleaned_response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)