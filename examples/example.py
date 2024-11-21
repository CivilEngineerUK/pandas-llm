import json
import pandas as pd
from src.pandas_query import PandasQuery

# Create sample DataFrame
df = pd.read_csv("customers-100.csv")

# Create query executor
querier = PandasQuery(validate=True, temperature=0)

# Execute query
try:
    # query = "Get a table of all customers who have a first name beginning with 'D' and who live in a city with exactly two e's in it?"
    # query = "Get a subtable of people who live in Panama"
    query = "Get a subtable of people whos surname backwards is: 'nosdodn' or 'atam'"
    result = querier.execute(df, query)

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

"""
Complete results:
{
  "query": "Get a subtable of people whos surname backwards is: 'nosdodn' or 'atam'",
  "code": "result = df[df['Last Name'].fillna('').str[::-1].str.lower().isin(['nosdodn', 'atam'])]",
  "is_valid": true,
  "errors": [],
  "result": [
    {
      "Index": 13,
      "Customer Id": "e35426EbDEceaFF",
      "First Name": "Tracey",
      "Last Name": "Mata",
      "Company": "Graham-Francis",
      "City": "South Joannamouth",
      "Country": "Togo",
      "Phone 1": "001-949-844-8787",
      "Phone 2": "(855)713-8773",
      "Email": "alex56@walls.org",
      "Subscription Date": "2021-12-02",
      "Website": "http://www.beck.com/"
    },
    {
      "Index": 18,
      "Customer Id": "F8Aa9d6DfcBeeF8",
      "First Name": "Greg",
      "Last Name": "Mata",
      "Company": "Valentine LLC",
      "City": "Lake Leslie",
      "Country": "Mozambique",
      "Phone 1": "(701)087-2415",
      "Phone 2": "(195)156-1861x26241",
      "Email": "jaredjuarez@carroll.org",
      "Subscription Date": "2022-03-26",
      "Website": "http://pitts-cherry.com/"
    }
  ]
}

Here is a table of the output results:

    Index      Customer Id  ... Subscription Date                   Website
12     13  e35426EbDEceaFF  ...        2021-12-02      http://www.beck.com/
17     18  F8Aa9d6DfcBeeF8  ...        2022-03-26  http://pitts-cherry.com/

[2 rows x 12 columns]
"""