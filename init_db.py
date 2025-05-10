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
            confidence REAL
        );
    """)
    conn.commit()
    conn.close()

# Jalankan hanya sekali saat setup awal
if __name__ == "__main__":
    create_database()
    print("✅ Database and table 'detections' created successfully.")
