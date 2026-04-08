"""
Session persistence layer for SupportSentinelEnv.
Stores session history and results to SQLite for recovery and analysis.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

DB_PATH = os.getenv("DB_PATH", "./data/sessions.db")

def init_database():
    """Initialize SQLite database for session persistence."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            seed INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active'
        )
    """)
    
    # Episode results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            steps INTEGER,
            final_score REAL,
            rewards TEXT,
            success BOOLEAN,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    
    # Metrics table for analytics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            step_number INTEGER,
            action_type TEXT,
            reward REAL,
            sentiment_score REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    
    conn.commit()
    conn.close()

def save_session(session_id: str, task_id: str, seed: int):
    """Save session to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR IGNORE INTO sessions (session_id, task_id, seed)
        VALUES (?, ?, ?)
    """, (session_id, task_id, seed))
    
    conn.commit()
    conn.close()

def save_result(session_id: str, task_id: str, steps: int, final_score: float, 
                rewards: List[float], success: bool):
    """Save episode result to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO results (session_id, task_id, steps, final_score, rewards, success)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, task_id, steps, final_score, json.dumps(rewards), success))
    
    conn.commit()
    conn.close()

def save_metric(session_id: str, step: int, action: str, reward: float, sentiment: float):
    """Save step-level metric."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO metrics (session_id, step_number, action_type, reward, sentiment_score)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, step, action, reward, sentiment))
    
    conn.commit()
    conn.close()

def get_session_history(session_id: str) -> Optional[Dict]:
    """Retrieve session history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT session_id, task_id, seed, created_at, status
        FROM sessions WHERE session_id = ?
    """, (session_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "session_id": row[0],
            "task_id": row[1],
            "seed": row[2],
            "created_at": row[3],
            "status": row[4]
        }
    return None

def get_task_statistics(task_id: str) -> Dict:
    """Get aggregate statistics for a task."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*), AVG(final_score), MAX(final_score), MIN(final_score)
        FROM results WHERE task_id = ?
    """, (task_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    return {
        "total_episodes": row[0] or 0,
        "avg_score": row[1] or 0.01,
        "max_score": row[2] or 0.01,
        "min_score": row[3] or 0.01
    }

# Initialize on import
init_database()
