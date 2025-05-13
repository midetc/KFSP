import sqlite3
import os

def get_db_path():
    return os.path.join(os.path.dirname(__file__), 'weather.sqlite3')

def init_db():
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS weather_forecast (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        min_temp REAL,
        max_temp REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def insert_forecast(df):
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    data = [(row['date'].strftime('%Y-%m-%d'), row['min_temp'], row['max_temp']) for _, row in df.iterrows()]
    cur.executemany('INSERT INTO weather_forecast (date, min_temp, max_temp) VALUES (?, ?, ?)', data)
    conn.commit()
    conn.close() 