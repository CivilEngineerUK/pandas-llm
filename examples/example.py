import pandas as pd
from src.pandas_query import PandasQuery

# Create sample DataFrame
df = pd.read_csv("customers-100.csv")

# Create query executor
querier = PandasQuery(validate=True)

# Execute query
try:
    result = querier.execute(df, "Get a table of all customers who have a first name beginning with 'D'?")

    # Get complete results as a dictionary
    result_dict = result.model_dump()  # Pydantic v2 syntax (or use .dict() for v1)
    print("\nComplete results:")
    import json
    print(json.dumps(result_dict, indent=2))

    # df of results
    print('\nHere is a table of the output results:\n')
    df_result = pd.DataFrame(result.result)
    print(df_result)

except Exception as e:
    print(f"Error executing query: {str(e)}")