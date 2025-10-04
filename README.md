# 🐄 BPA Smart Breed Identifier

**AI-Powered Cattle & Buffalo Breed Classification for Field Level Workers (FLWs)**  

[🎥 **Watch Demo**](https://youtu.be/azF_eOF1YCc?si=lQe1HxNkiksjX_Zs)

---

## 📋 Table of Contents

- [Overview](#overview)  
- [Problem Statement](#problem-statement)  
- [Key Features](#key-features)  
- [Technology Stack](#technology-stack)  
- [System Architecture](#system-architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Supported Breeds](#supported-breeds)  
- [Analytics Dashboard](#analytics-dashboard)  
- [Impact & Benefits](#impact--benefits)  
- [Future Scope](#future-scope)  
- [Disclaimer](#disclaimer)  
- [Developed During SIH 2025](#developed-during-sih-2025)  
- [Conclusion](#conclusion)  

---

## 🌟 Overview

**BPA Smart Breed Identifier** is an AI-powered livestock management system designed to assist Field Level Workers (FLWs) in accurately identifying cattle and buffalo breeds during BPA (Bharat Pashudhan App) data collection.  

By combining **deep learning-based breed recognition** with a **smart registration workflow**, the system reduces misclassification errors, enhances data quality, and supports evidence-based decision-making for livestock programs.  

---

## 🚨 Problem Statement

Manual breed identification often leads to:

- 📊 **High misclassification rates**  
- 🗂️ **Poor data quality** for research and policy planning  
- 💰 **Inefficient resource allocation**  
- 🧬 **Suboptimal genetic improvement outcomes**  
- ⏱️ **Time-consuming training** for FLWs  
- 📱 **Limited field support** in remote areas  

---

## ✨ Key Features

### 🔍 AI Breed Recognition
- Deep learning-based breed classification using **MobileNetV2**  
- Real-time image analysis and multiple prediction suggestions  
- Manual override option for high accuracy  

### 🎯 Smart Registration Workflow
- Instant breed suggestions for FLWs  
- BPA-compatible data collection and validation  
- User-friendly interface optimized for field conditions  

### 📊 Analytics Dashboard
- Real-time breed distribution and registration statistics  
- Geographic mapping of livestock populations  
- Production and performance analysis  

### 🌍 Multi-language Support
- **11+ Indian languages** including Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu, English  

### 🤖 AI-Powered Breed Information
- Comprehensive breed profiles with key identification features  
- Production metrics, management requirements, and economic importance  

### 📺 Educational Resources
- Categorized YouTube learning content  
- Training modules and quick reference guides for FLWs  

---

## 🛠 Technology Stack

**AI/ML:** TensorFlow, Keras, MobileNetV2, NumPy, Pillow  
**Backend:** Streamlit, Phi Agent Framework, Groq API (Llama 3.3 70B)  
**Data Visualization:** Plotly Express, Pandas  
**Integration:** YouTube Search API, Base64 encoding  
**Deployment:** Python 3.8+, Streamlit Cloud-ready  

---

## 🏗 System Architecture

📷 Animal Image
│
▼
🧠 AI Model (MobileNetV2)
│
┌────┴────┐
▼ ▼
🎯 Breed Suggestion 🤖 Breed Info (Groq LLM)
│ │
▼ ▼
📋 BPA Registration & Data Collection
│
▼
📊 Analytics Dashboard

yaml
Copy code

---

## 💻 Installation

### Prerequisites
- Python 3.8+  
- pip package manager  
- Virtual environment recommended  
- Internet connection for initial setup  

### Clone Repository
```bash
git clone https://github.com/ProxyCode1010/AI-Breed-Classifier-for-Bharat-Pashudhan-App.git
cd AI-Breed-Classifier-for-Bharat-Pashudhan-App
```
Setup Virtual Environment
bash
Copy code
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Configure API Keys
Create a .env file in the project root:

bash
Copy code
GROQ_API_KEY=your_groq_api_key_here
▶️ Usage
Launch the application:

bash
Copy code
streamlit run app.py
Select language from sidebar.

Upload a clear animal image (JPG/PNG).

Review AI breed suggestion and alternatives.

Confirm or manually select breed.

Complete BPA registration (owner info, farm details, animal metrics, production data).

Access educational resources and YouTube training content.

📊 Analytics Dashboard
Real-time breed distribution and registration tracking

Geographic distribution (state/district level)

Production analysis (milk yield, age correlations)

Exportable registration records

🌍 Impact & Benefits
✅ Improved data quality & reduced breed misclassification

✅ Faster, efficient animal registration

✅ Better resource allocation & breeding program effectiveness

✅ Empowered FLWs with AI assistance and multi-language support

🚀 Future Scope
Video-based breed identification

Multi-angle verification & age estimation

Mobile apps with offline mode

Direct BPA server integration & veterinary services

Cloud-based deployment & API for third-party integration

## ⚠️ Disclaimer

> **Note:** Developed for educational and demonstration purposes (SIH 2025).  
> AI suggestions **should not replace human verification**. Manual confirmation by trained FLWs or veterinary experts is mandatory.  
> The developers are **not responsible** for any consequences arising from use of AI suggestions.


🎓 Developed During SIH 2025
Problem ID: 25004
Ministry: Ministry of Fisheries, Animal Husbandry & Dairying

This project was created as part of Smart India Hackathon 2025 for image-based breed recognition of cattle and buffaloes in India.

✅ Conclusion
BPA Smart Breed Identifier combines AI, mobile technology, and multi-language support to:

Assist FLWs in breed identification

Reduce data errors

Enable faster and smarter BPA registration

Support better decision-making and livestock management

Made with 🐄 for sustainable livestock management and rural development
