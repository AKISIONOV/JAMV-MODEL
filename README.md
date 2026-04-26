# 🤖 JAMV MODEL - Full Stack ML Web Application

A complete full-stack web application with a machine learning model, FastAPI backend, and modern React-like frontend.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)

## 📁 Project Structure

```
JAMV MODEL/
├── best_model.keras          # Trained ML model
├── backend/
│   ├── app.py               # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Modern web interface
├── README.md                # This file
└── .gitignore               # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd "JAMV MODEL"
```

2. **Create virtual environment:**
```bash
python -m venv .venv
```

3. **Activate virtual environment:**

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r backend/requirements.txt
```

### Running the Application

1. **Start the backend API:**
```bash
cd backend
python app.py
```

The API will be available at `http://localhost:8000`

2. **Open the frontend:**
Open `frontend/index.html` in your browser, or serve it:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 3000
```

Then open `http://localhost:3000` in your browser.

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/predict` | POST | Make single prediction |
| `/predict/batch` | POST | Make batch predictions |

## 📝 Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 3.4, 5.6, 2.1, 0.8, 4.5, 3.2, 1.9]}'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"requests": [{"features": [1.2, 3.4]}, {"features": [5.6, 2.1]}]}'
```

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **TensorFlow/Keras** - ML model loading and inference
- **Uvicorn** - ASGI server

### Frontend
- **Vanilla JavaScript** - No framework dependencies
- **Modern CSS** - Custom properties, Grid, Flexbox
- **Fetch API** - Async HTTP requests

## 📄 License

MIT License - feel free to use this project for any purpose.

---

**Built with ❤️ using FastAPI + TensorFlow**