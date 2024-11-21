import json
import pandas as pd
from src.pandas_query import PandasQuery

# Create sample DataFrame
df = pd.read_csv("customers-100.csv")

# Create query executor
querier = PandasQuery(validate=True, temperature=0.2)

# Execute query
try:
    result = querier.execute(df, "Get a table of all customers who have a first name beginning with 'D' and who live in a city with exactly two e's in it?")

    # Get complete results as a dictionary
    result_dict = result.model_dump()
    print("\nComplete results:")
    print(json.dumps(result_dict, indent=2))

    # df of results
    print('\nHere is a table of the output results:\n')
    df_result = pd.DataFrame(result.result)
    print(df_result)

except Exception as e:
    print(f"Error executing query: {str(e)}")