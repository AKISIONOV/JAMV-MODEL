import sqlite3
import os
from datetime import datetime
import json

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "predictions.db")

def init_db():
    """Initialize the SQLite database and create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        predicted_class INTEGER NOT NULL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        all_probabilities TEXT NOT NULL,
        severity TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction(filename, predicted_class, class_name, confidence, all_probs, severity, recommendation):
    """Save a prediction result to the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        probs_json = json.dumps(all_probs)
        
        cursor.execute('''
        INSERT INTO predictions 
        (filename, predicted_class, class_name, confidence, all_probabilities, severity, recommendation, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (filename, predicted_class, class_name, confidence, probs_json, severity, recommendation, timestamp))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return prediction_id
    except Exception as e:
        print(f"Error saving prediction to DB: {e}")
        return None

def get_prediction_history(limit=50):
    """Retrieve recent predictions from the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        
        history = []
        for row in rows:
            record = dict(row)
            record['all_probabilities'] = json.loads(record['all_probabilities'])
            history.append(record)
            
        conn.close()
        return history
    except Exception as e:
        print(f"Error retrieving prediction history: {e}")
        return []
