import sqlite3

def create_database():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame INTEGER,
            photo TEXT,
            class TEXT,
            date TEXT,
            confidence REAL,
            clahe INTEGER  -- 0 = tanpa CLAHE, 1 = dengan CLAHE
        );
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    print("✅ Database and table 'detections' created successfully.")
