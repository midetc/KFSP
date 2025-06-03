import sqlite3


def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER NOT NULL,
            avg_fitness REAL NOT NULL,
            best_fitness REAL NOT NULL,
            best_individual REAL NOT NULL
        )
    ''')

    cursor.execute('DELETE FROM runs')

    conn.commit()
    conn.close()


def save_stats_to_db(db_path, generation, avg_fitness,
                     best_fitness, best_individual):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO runs (generation, avg_fitness, best_fitness,
                         best_individual)
        VALUES (?, ?, ?, ?)
    ''', (generation, avg_fitness, best_fitness, best_individual))
    conn.commit()
    conn.close()


def get_all_runs(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT generation, avg_fitness, best_fitness, best_individual
        FROM runs
        ORDER BY generation
    ''')

    runs = cursor.fetchall()
    conn.close()

    return runs
