# ChatBot Excel Data Analyzer

An AI-powered chatbot that analyzes Excel data using Google's Gemini API. Built with React and Flask.

## Setup & Installation

### Backend Setup

1. Navigate to backend folder:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
```

3. Activate virtual environment:
```bash
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install flask flask-cors python-dotenv pandas openpyxl google-generativeai
```

5. Create .env file in backend folder:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

6. Run backend server:
```bash
python app.py
```
Server will run on http://localhost:5001

### Frontend Setup

1. Navigate to frontend folder:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```
Frontend will run on http://localhost:3000

## Usage

1. Open http://localhost:3000 in your browser
2. Upload an Excel file using the upload button
3. Start chatting with the AI about your data
4. Get instant analysis and insights about your Excel data

## Features

- Excel file upload and validation
- Interactive chat interface
- Real-time AI analysis
- Particle background effects
- Responsive design

## Tech Stack

- Frontend: React, TailwindCSS, Particles.js
- Backend: Flask, Python
- AI: Google Gemini API
- Data Processing: Pandas