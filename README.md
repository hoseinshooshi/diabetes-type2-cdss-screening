🩺 Diabetes CDSS — AI-Powered Clinical Decision Support System

This repository contains a Flask-based Clinical Decision Support System (CDSS) for type 2 diabetes risk prediction and personalized recommendations, developed with AI assistance.
The app combines machine learning (Random Forest) and DeepSeek AI to deliver intelligent, data-driven medical insights based on patient features like glucose, blood pressure, BMI, and age.

🚀 Features

Flask REST API for diabetes risk prediction

DeepSeek AI integration for generating personalized Persian-language medical advice

Random Forest model trained on the PIMA Indians Diabetes dataset

Rule-based logic layer for refining predictions

AI-enhanced analysis and recommendations

CORS-enabled for smooth frontend integration

🧠 Tech Stack

Backend: Flask, scikit-learn, NumPy, Pandas

AI Layer: DeepSeek API (via OpenAI client)

Model: Random Forest with StandardScaler preprocessing

Frontend: React (or any modern JS framework consuming the Flask API)

⚙️ Endpoints
Method	Endpoint	Description
POST	/predict	Predict diabetes risk and get AI-driven recommendations
GET	/health	Check service and model health
GET	/	API root endpoint with system info
🧩 How It Works

User submits patient data via /predict.

The Random Forest model predicts diabetes risk.

Rule-based refinements adjust the result.

DeepSeek AI provides a natural-language analysis and advice.

The API returns risk level, recommendations, and medical alerts.

🩸 Dataset

Trained using the PIMA Indians Diabetes dataset:
👉 https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

💡 Purpose

Designed as an AI-assisted medical decision support system for early diabetes screening, clinical training, or integration into telemedicine tools.
