import sqlite3
import pandas as pd
import json

# Connect to SQLite database
conn = sqlite3.connect('sql/madness2025.db')

# Load data into a DataFrame
query = "SELECT * FROM TeamGameStats"
df = pd.read_sql_query(query, conn)

# Select only numeric columns
numeric_cols = df.select_dtypes(include='number').columns

# Compute mean and standard deviation
stats_summary = df[numeric_cols].agg(['mean', 'std']).transpose()

# Convert to the requested JSON structure
stats_json = stats_summary.to_dict(orient='index')

# Save to a JSON file
with open('real_stats_distributions.json', 'w') as f:
    json.dump(stats_json, f, indent=4)

# Close connection
conn.close()

print(stats_json)