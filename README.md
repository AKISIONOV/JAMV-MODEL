<div align="center">

# 🩺 JAMV MODEL: Edge AI Diabetic Retinopathy Detection

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-FF6F00?logo=tensorflow&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![SQLite](https://img.shields.io/badge/SQLite-07405e?style=flat&logo=sqlite&logoColor=white)](#)
[![Vanilla JS](https://img.shields.io/badge/Vanilla_JS-F7DF1E?style=flat&logo=javascript&logoColor=black)](#)

*A full-stack, production-ready machine learning application designed to instantly predict Diabetic Retinopathy severity from retinal images using highly optimized Edge AI.*

[Key Features](#-key-features) • [Tech Stack](#%EF%B8%8F-technology-stack) • [Installation](#-quick-start) • [API Reference](#-api-endpoints)

</div>

---

## 🌟 Overview

The **JAMV MODEL** is a state-of-the-art diagnostic tool aimed at assisting medical professionals in the early detection of Diabetic Retinopathy. 

Instead of relying on massive, slow server-side models, this project utilizes **TensorFlow Lite Quantization (Float16)** to compress a complex Deep Learning model from 20MB down to **3.4MB**—resulting in lightning-fast inference times without sacrificing clinical accuracy. The prediction results are securely tracked using an embedded **SQLite** database, all wrapped in a premium, responsive web interface.

---

## ✨ Key Features

- ⚡ **Optimized Edge AI Inference**: Uses a quantized `.tflite` model (Float16) to achieve high-speed inference with a 6x reduction in model size.
- 🎨 **Premium UI/UX**: A modern "Medical Blue" interface built with CSS Variables, featuring glassmorphism, dynamic animations, and drag-and-drop image uploads.
- 🗄️ **Persistent Prediction History**: Every diagnosis is automatically saved to an SQLite database (`predictions.db`) for auditing and historical tracking.
- 📊 **Clinical Reporting**: Built-in visual dashboards showcasing model Accuracy, Loss, Confusion Matrices, and Class Distribution.
- 🔌 **RESTful API**: A robust FastAPI backend capable of handling single and batch prediction requests.

---

## 🛠️ Technology Stack

### Machine Learning & Backend
* **Python 3.13+**
* **TensorFlow Lite**: For loading the `.tflite` quantized model and running rapid predictions.
* **FastAPI & Uvicorn**: Delivering a high-performance, asynchronous web server.
* **SQLite3**: Lightweight, file-based database for tracking historical predictions.
* **Pillow & NumPy**: For efficient image preprocessing and tensor manipulation (resizing to strictly enforced `100x100` input tensors).

### Frontend
* **Vanilla JavaScript (ES6)**: No bloat, pure DOM manipulation and Fetch API integration.
* **CSS3 (Custom Properties & Grid)**: Fully responsive, scalable design system with smooth micro-animations.

---

## 🚀 Quick Start

Follow these steps to run the application locally on your machine.

### Prerequisites
- Python 3.13 or higher
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AKISIONOV/JAMV-MODEL.git
cd "JAMV-MODEL"
```

2. **Create a virtual environment:**
```bash
python -m venv .venv313
```

3. **Activate the virtual environment:**

*Windows:*
```cmd
.venv313\Scripts\activate
```
*macOS/Linux:*
```bash
source .venv313/bin/activate
```

4. **Install backend dependencies:**
```bash
pip install -r backend/requirements.txt tensorflow
```

### Running the Application

1. **Start the FastAPI Backend Server:**
```bash
cd backend
python app.py
```
*(The API will be live at `http://localhost:8000`)*

2. **Open the Frontend Interface:**
Simply navigate to the `frontend` directory and open `index.html` in your web browser. 
*(You can just double-click the file!)*

---

## 🩺 Diagnosis Categories

The model classifies retinal images into 5 distinct Diabetic Retinopathy severity levels:

1. **No DR** (No Diabetic Retinopathy)
2. **Mild NPDR** (Non-Proliferative DR)
3. **Moderate NPDR**
4. **Severe NPDR**
5. **Proliferative DR** (PDR)

*The API also returns automated clinical recommendations based on the severity.*

---

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Root API status and system checks |
| `/health` | `GET` | Verifies if the TFLite model is correctly loaded into memory |
| `/predict/image` | `POST` | Accepts a multipart/form-data image upload and returns the DR diagnosis |
| `/predictions` | `GET` | Retrieves the latest prediction records from the SQLite database |
| `/classes` | `GET` | Returns the list of supported diagnosis severity classes |

---

<div align="center">
  <b>Built with ❤️ using FastAPI and TensorFlow Lite</b>
</div>