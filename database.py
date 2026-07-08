import sqlite3

conn = sqlite3.connect("emails.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    prediction TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully.")