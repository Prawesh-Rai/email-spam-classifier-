import sqlite3

conn = sqlite3.connect("emails.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM predictions")

rows = cursor.fetchall()

print("\nStored Predictions:\n")

for row in rows:
    print(row)

conn.close()