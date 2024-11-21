# examples/example.py
import os
import pandas as pd
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.pandas_query import PandasQuery

# Data
data = [
    ('John Doe', 25, 50),
    ('Jane Smith', 38, 70),
    ('Alex Johnson', 45, 80),
    ('Jessica Brown', 60, 40),
    ('Michael Davis', 22, 90),
]
df = pd.DataFrame(data, columns=['name', 'age', 'donation'])

# Create query executor
querier = PandasQuery()

# Execute query
query = "What is the average donation of people older than 40 who donated more than $50?"
result = querier.execute(df, query)

print(f"Query: {query}")
print(f"Generated code: {querier.last_code}")
print(f"Result: {result}")