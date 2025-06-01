# starting code of excel without chroma
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import re
import logging
import json

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

    allowed_extensions = ('.xlsx', '.xls', '.csv', '.json')
    if not file.filename.lower().endswith(allowed_extensions):
        return jsonify({"error": "Please upload only Excel (.xlsx, .xls), CSV (.csv), or JSON (.json) files."}), 400

    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif file.filename.lower().endswith('.json'):
            file.seek(0)
            try:
                # Try to load as a flat table
                df = pd.read_json(file)
            except Exception:
                # Try to flatten nested JSON
                file.seek(0)
                raw = json.load(file)
                try:
                    # If it's a dict, try to flatten each list value
                    if isinstance(raw, dict):
                        # Try to flatten each list in the dict
                        for v in raw.values():
                            if isinstance(v, list):
                                df = pd.json_normalize(v)
                                break
                        else:
                            # If no list found, flatten the dict itself
                            df = pd.json_normalize(raw)
                    elif isinstance(raw, list):
                        df = pd.json_normalize(raw)
                    else:
                        return jsonify({"error": "Unsupported JSON structure."}), 400
                except Exception:
                    return jsonify({"error": "Could not flatten JSON. Please upload a tabular or supported nested JSON."}), 400
        else:
            return jsonify({"error": "Unsupported file type."}), 400

        if df.empty:
            return jsonify({"error": "Uploaded file contains no data."}), 400

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
        return jsonify({"response": "Please upload an Excel file with at least 500 records first."})

    df = excel_data['df']
    columns = df.columns.tolist()
    columns_lower = [col.lower() for col in columns]

    # 1. Calculation detection (sum, average, count, min, max)
    calc_patterns = [
        (r'(average|mean|avg) of ([\w\s]+)', 'mean'),
        (r'(average|mean|avg) ([\w\s]+)', 'mean'),
        (r'calculate (average|mean|avg) of ([\w\s]+)', 'mean'),
        (r'calculate (average|mean|avg) ([\w\s]+)', 'mean'),
        (r'sum of ([\w\s]+)', 'sum'),
        (r'sum ([\w\s]+)', 'sum'),
        (r'count of ([\w\s]+)', 'count'),
        (r'count ([\w\s]+)', 'count'),
        (r'min(?:imum)? of ([\w\s]+)', 'min'),
        (r'min(?:imum)? ([\w\s]+)', 'min'),
        (r'max(?:imum)? of ([\w\s]+)', 'max'),
        (r'max(?:imum)? ([\w\s]+)', 'max'),
    ]
    for pattern, calc_type in calc_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            if match.lastindex == 2:
                col_query = match.group(2).strip().lower()
            else:
                col_query = match.group(1).strip().lower()
            for col in columns:
                if col.lower() == col_query:
                    try:
                        numeric_series = pd.to_numeric(df[col], errors='coerce').dropna()
                        if numeric_series.empty:
                            answer = f"No valid numeric values found in '{col}'."
                        elif calc_type == 'mean':
                            result = numeric_series.mean()
                            sum_ = numeric_series.sum()
                            count_ = numeric_series.count()
                            answer = (
                                f"The average of '{col}' is {result:.2f}. "
                                f"(Validation: sum={sum_}, count={count_}, calculated average={sum_/count_:.2f})"
                            )
                        elif calc_type == 'sum':
                            result = numeric_series.sum()
                            count_ = numeric_series.count()
                            answer = (
                                f"The sum of '{col}' is {result}. "
                                f"(Validation: sum={result}, count={count_})"
                            )
                        elif calc_type == 'count':
                            result = df[col].count()
                            answer = (
                                f"The count of '{col}' is {result}. "
                                f"(Validation: count={result})"
                            )
                        elif calc_type == 'min':
                            result = numeric_series.min()
                            answer = (
                                f"The minimum of '{col}' is {result}. "
                                f"(Validation: min={result}, total_values={numeric_series.count()})"
                            )
                        elif calc_type == 'max':
                            # If you want to ignore negative values, uncomment the next line:
                            # numeric_series = numeric_series[numeric_series >= 0]
                            result = numeric_series.max()
                            answer = (
                                f"The maximum of '{col}' is {result}. "
                                f"(Validation: max={result}, total_values={numeric_series.count()})"
                            )
                    except Exception as e:
                        answer = f"Could not calculate {calc_type} for '{col}': {str(e)}"
                    # Return the answer directly, skip LLM
                    return jsonify({"response": answer})

    # 2. Try to extract column-value pairs from the user message (as before)
    found = False
    result_rows = pd.DataFrame()
    for col in columns:
        pattern1 = rf"{col}\s*(is|=|:)?\s*([a-zA-Z0-9@.\-_ ]+)"
        pattern2 = rf"with\s+{col}\s*([a-zA-Z0-9@.\-_ ]+)"
        match1 = re.search(pattern1, user_message, re.IGNORECASE)
        match2 = re.search(pattern2, user_message, re.IGNORECASE)
        value = None
        if match1:
            value = match1.group(2).strip()
        elif match2:
            value = match2.group(1).strip()
        if value:
            mask = df[col].astype(str).str.lower().str.strip() == value.lower()
            result_rows = df[mask]
            if not result_rows.empty:
                found = True
                break

    if found:
        sample = result_rows.head(3).to_dict(orient='records')
        system_prompt = (
            "You are a data assistant. Present the following data to the user in a clear, friendly way. "
            "If there are multiple matches, summarize them. Do not use '#' or '###' for headings."
        )
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {user_message}\nMatching data: {sample}"}
            ],
            extra_headers=EXTRA_HEADERS
        )
        cleaned_response = response.choices[0].message.content.strip()
        return jsonify({"response": cleaned_response})

    # 3. Fallback: send schema and sample as before
    schema = df.dtypes.astype(str).to_dict()
    sample_head = df.head(5).to_dict(orient='records')
    sample_tail = df.tail(5).to_dict(orient='records')
    sample_text = (
        f"Columns: {columns}\n"
        f"Schema: {schema}\n"
        f"Sample (first 5 rows): {sample_head}\n"
        f"Sample (last 5 rows): {sample_tail}\n"
        f"Total records: {len(df)}"
    )
    system_prompt = """You are a data analyst assistant. 
You must answer any question that is relevant to the uploaded Excel dataset, using only the information from the dataset's schema and provided samples. 
If a question is not related to the dataset, respond with: 'I can only answer questions relevant to the uploaded dataset.' 
If you need more data to answer, ask the user to clarify or request a specific sample.
Always provide clear, well-formatted answers using paragraphs and bullet points where helpful. 
Do not use '#' or '###' for headings."""

    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Given this Excel data sample:
{sample_text}

User question: {user_message}

Please analyze the data and provide a clear, well-formatted response."""}
        ],
        extra_headers=EXTRA_HEADERS
    )
    cleaned_response = response.choices[0].message.content.replace('###', '').replace('#', '').strip()
    return jsonify({"response": cleaned_response})


if __name__ == '__main__':
    app.run(debug=True)