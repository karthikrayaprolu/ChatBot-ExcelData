from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import re
import logging
import json
import chromadb
from sentence_transformers import SentenceTransformer

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

# Initialize Chroma client and embedding model
chroma_client = chromadb.Client()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "excel_data"

# For tracking last calculation context
last_calc_context = {}

def upsert_to_chroma(df):
    def flatten_metadata(row):
        meta = {}
        for k, v in row.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                meta[k] = v
            else:
                # Convert lists/dicts/other to JSON string
                try:
                    meta[k] = json.dumps(v)
                except Exception:
                    meta[k] = str(v)
        return meta

    texts = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    embeddings = embedding_model.encode(texts)
    ids = [str(i) for i in range(len(texts))]
    metadatas = [flatten_metadata(row) for row in df.to_dict(orient="records")]
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

def query_chroma(query, top_k=5):
    collection = chroma_client.get_collection(COLLECTION_NAME)
    query_embedding = embedding_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return results

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
                df = pd.read_json(file)
            except Exception:
                file.seek(0)
                raw = json.load(file)
                # Try to flatten SQuAD-style JSON
                if isinstance(raw, dict) and 'data' in raw:
                    df = flatten_squad_json(raw)
                # Try to flatten nested JSON
                file.seek(0)
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
        upsert_to_chroma(df)  # Store in Chroma
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

    # Patterns for summary/overview questions
    summary_patterns = [
        r"\b(what (does|do) (it|this|the data(set)?) contain)\b",
        r"\b(what is (it|this|the data(set)?) about)\b",
        r"\b(describe|summary|summarize|overview|structure|info|information)\b",
        r"\bwhat (it|this|the data(set)?) has\b",
        r"\bwhat columns\b",
        r"\bshow columns\b",
        r"\bwhich columns\b",
        r"\bcolumn names\b",
        r"\boverview\b",
        r"\babout\b"
    ]
    if any(re.search(pat, user_message, re.IGNORECASE) for pat in summary_patterns):
        columns = df.columns.tolist()
        num_rows = len(df)
        num_cols = len(columns)
        col_types = [f"**{col}** (`{str(df[col].dtype)}`)" for col in columns]
        # Optionally, try to guess column purposes (very basic)
        col_descriptions = []
        for col in columns:
            dtype = str(df[col].dtype)
            if "int" in dtype or "float" in dtype:
                desc = f"- 🔢 **{col}**: Numeric column, likely representing quantities or measurements."
            elif "date" in dtype or "time" in dtype:
                desc = f"- 📅 **{col}**: Contains date or time information."
            else:
                desc = f"- 🏷️ **{col}**: Categorical or text column."
            col_descriptions.append(desc)
        summary = (
            f"## 📊 Dataset Overview\n"
            f"- **Rows:** {num_rows}\n"
            f"- **Columns:** {num_cols}\n"
            f"- **Column List:** {', '.join(col_types)}\n\n"
            f"### Column Details:\n"
            f"{chr(10).join(col_descriptions)}\n\n"
            f"💡 *You can ask about the contents, request statistics or calculations for numeric columns, "
            f"or inquire about specific values or patterns in the data!*"
        )
        return jsonify({"response": summary})

    # Patterns for calculation queries
    calc_patterns = [
        (r'(average|mean|avg) of ([\w\s]+)', 'mean'),
        (r'(average|mean|avg) ([\w\s]+)', 'mean'),
        (r'sum of ([\w\s]+)', 'sum'),
        (r'sum ([\w\s]+)', 'sum'),
        (r'count of ([\w\s]+)', 'count'),
        (r'count ([\w\s]+)', 'count'),
        (r'min(?:imum)? of ([\w\s]+)', 'min'),
        (r'min(?:imum)? ([\w\s]+)', 'min'),
        (r'max(?:imum)? of ([\w\s]+)', 'max'),
        (r'max(?:imum)? ([\w\s]+)', 'max'),
    ]
    # Add this pattern to detect validation/explanation requests
    validation_patterns = [
        r"\b(show|explain|how|validation|steps|process|prove|demonstrate)\b"
    ]
    general_validation_patterns = [
        r"\b(validate|validation|prove|show steps|how did you get|how was it calculated)\b"
    ]

    for pattern, calc_type in calc_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            col = match.group(2).strip() if match.lastindex >= 2 else match.group(1).strip()
            # Try to match column name (case-insensitive)
            col_match = next((c for c in df.columns if c.lower() == col.lower()), None)
            if not col_match:
                # Try partial match
                col_match = next((c for c in df.columns if col.lower() in c.lower()), None)
            if not col_match:
                return jsonify({"response": f"❌ The column '{col}' does not exist in the dataset."})

            # Check if column has data
            if df[col_match].dropna().empty:
                return jsonify({"response": f"⚠️ The column '{col_match}' has no data to calculate."})

            # Check if column is numeric for relevant operations
            is_numeric = pd.api.types.is_numeric_dtype(df[col_match])
            if calc_type in ['mean', 'sum', 'min', 'max'] and not is_numeric:
                return jsonify({"response": f"❌ The column '{col_match}' is not numeric, so {calc_type} cannot be calculated."})

            # Detect if user wants validation/explanation
            wants_validation = any(re.search(pat, user_message, re.IGNORECASE) for pat in validation_patterns)

            try:
                if calc_type == 'mean':
                    value = df[col_match].mean()
                    if wants_validation:
                        resp = (
                            f"**How the average (mean) of '{col_match}' was calculated:**\n"
                            f"1. All non-empty values in the column were selected.\n"
                            f"2. The sum of these values was divided by their count.\n"
                            f"3. Formula: mean = sum(values) / count(values)\n"
                            f"4. Result: mean = {df[col_match].sum()} / {df[col_match].count()} = **{value:.2f}**"
                        )
                    else:
                        resp = f"🔎 The **average** of `{col_match}` is **{value:.2f}**."
                elif calc_type == 'sum':
                    value = df[col_match].sum()
                    if wants_validation:
                        resp = (
                            f"**How the sum of '{col_match}' was calculated:**\n"
                            f"1. All non-empty values in the column were selected.\n"
                            f"2. All values were added together.\n"
                            f"3. Result: sum = **{value}**"
                        )
                    else:
                        resp = f"➕ The **sum** of `{col_match}` is **{value}**."
                elif calc_type == 'count':
                    value = df[col_match].count()
                    if wants_validation:
                        resp = (
                            f"**How the count of '{col_match}' was calculated:**\n"
                            f"1. All non-empty values in the column were counted.\n"
                            f"2. Result: count = **{value}**"
                        )
                    else:
                        resp = f"🔢 The **count** of `{col_match}` is **{value}**."
                elif calc_type == 'min':
                    value = df[col_match].min()
                    if wants_validation:
                        resp = (
                            f"**How the minimum of '{col_match}' was calculated:**\n"
                            f"1. All non-empty values in the column were selected.\n"
                            f"2. The smallest value was chosen.\n"
                            f"3. Result: min = **{value}**"
                        )
                    else:
                        resp = f"🔽 The **minimum** value in `{col_match}` is **{value}**."
                elif calc_type == 'max':
                    value = df[col_match].max()
                    if wants_validation:
                        resp = (
                            f"**How the maximum of '{col_match}' was calculated:**\n"
                            f"1. All non-empty values in the column were selected.\n"
                            f"2. The largest value was chosen.\n"
                            f"3. Result: max = **{value}**"
                        )
                    else:
                        resp = f"🔼 The **maximum** value in `{col_match}` is **{value}**."
                else:
                    resp = "❓ Sorry, I couldn't perform the requested calculation."
                last_calc_context['type'] = calc_type
                last_calc_context['column'] = col_match
                last_calc_context['value'] = value
                last_calc_context['count'] = df[col_match].count()
                last_calc_context['sum'] = df[col_match].sum() if calc_type == 'mean' else None
                return jsonify({"response": resp})
            except Exception as e:
                return jsonify({"response": f"⚠️ Sorry, there was an error calculating {calc_type} for '{col_match}': {str(e)}"})

    # Detect validation request
    if any(re.search(pat, user_message, re.IGNORECASE) for pat in validation_patterns + general_validation_patterns):
        if last_calc_context.get('type') == 'count':
            col = last_calc_context['column']
            value = last_calc_context['value']
            resp = (
                f"**Validation for count of '{col}':**\n"
                f"1. All non-empty values in the column '{col}' were counted.\n"
                f"2. Result: count = **{value}**"
            )
            return jsonify({"response": resp})
        elif last_calc_context.get('type') == 'mean':
            col = last_calc_context['column']
            value = last_calc_context['value']
            total = last_calc_context['sum']
            count = last_calc_context['count']
            resp = (
                f"**Validation for average (mean) of '{col}':**\n"
                f"1. All non-empty values in the column '{col}' were selected.\n"
                f"2. The sum of these values is {total}.\n"
                f"3. The count of these values is {count}.\n"
                f"4. The mean is calculated as sum/count = {total}/{count} = **{value:.2f}**"
            )
            return jsonify({"response": resp})
        # Add similar blocks for sum, min, max if needed
        else:
            return jsonify({"response": "⚠️ No recent calculation to validate. Please ask for a calculation first."})

    # Query Chroma for relevant rows
    chroma_results = query_chroma(user_message, top_k=5)
    context_rows = chroma_results['documents'][0] if chroma_results['documents'] else []
    context_text = "\n".join(context_rows)

    # Fallback: If no relevant rows found, provide a summary
    if not context_rows:
        columns = df.columns.tolist()
        sample = df.head(3).to_dict(orient="records")
        summary = (
            f"Sorry, I couldn't find a direct answer to your question in the data. "
            f"The dataset contains {len(df)} rows and the following columns: {', '.join(columns)}. "
            f"Here are a few sample rows: {json.dumps(sample, indent=2)}"
        )
        return jsonify({"response": summary})

    # Use context_text in your LLM prompt
    system_prompt = (
        "You are a data assistant. Use the following relevant data rows to answer the user's question. "
        "If the answer is not in the data, say so. Respond in clear, plain English."
    )
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User question: {user_message}\nRelevant data:\n{context_text}"}
        ],
        extra_headers=EXTRA_HEADERS
    )
    cleaned_response = response.choices[0].message.content.strip()
    return jsonify({"response": cleaned_response})

def flatten_squad_json(raw):
    rows = []
    for entry in raw['data']:
        title = entry.get('title', '')
        for para in entry.get('paragraphs', []):
            context = para.get('context', '')
            for qa in para.get('qas', []):
                question = qa.get('question', '')
                answer = qa.get('answers', [{}])[0].get('text', '')
                rows.append({
                    'title': title,
                    'context': context,
                    'question': question,
                    'answer': answer
                })
    return pd.DataFrame(rows)

if __name__ == '__main__':
    app.run(debug=True)