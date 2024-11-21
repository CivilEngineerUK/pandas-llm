# pandas-LLM

## Introduction
pandas-llm is a lightweight Python library that extends pandas to allow querying datasets using OpenAI prompts. This powerful tool leverages the natural language processing capabilities of OpenAI to offer intuitive, language-based querying of your Pandas dataframes with built-in validation and safety features.

## Key Features
- **Natural Language Querying**: Execute complex Pandas queries using natural language prompts. Instead of writing code, express your query in plain language and obtain the desired results.

- **Data Privacy**: Your data stays local. Pandas-LLM works with your data locally and uses OpenAI to create queries based on dataframe metadata (columns and data types), not its content.

- **Query Validation**: Built-in validation ensures generated queries are safe and compatible with your data types, preventing common errors and ensuring reliable results.

- **Safe Execution**: Uses RestrictedPython for sandboxed execution of generated queries, providing an additional layer of security.

- **Serializable Results**: Results are automatically converted to JSON-serializable formats, making it easy to store or transmit query results.

- **Type-Safe Operations**: Intelligent handling of different data types including strings, numbers, dates, and boolean values with appropriate null value handling.

## Installation

Install pandas-llm using pip:

```shell
pip install pandas-llm
```

## Usage
Here's a basic example of how to use pandas-llm:

```python
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
```

## Query Result Structure
The library returns a QueryResult object with the following attributes:

```python
{
    "query": str,          # Original natural language query
    "code": str,          # Generated pandas code
    "is_valid": bool,     # Whether the query passed validation
    "errors": List[str],  # Any validation or execution errors
    "result": Any         # Query results (automatically serialized)
}
```

## Supported Operations
The library supports a wide range of pandas operations:

### String Operations
- Basic: contains, startswith, endswith, lower, upper, strip
- Count/Match: count, match, extract, find, findall
- Transform: replace, pad, center, slice, split

### Numeric Operations
- Comparisons: >, <, >=, <=, ==, !=
- Aggregations (with groupby): sum, mean, median, min, max, count

### Date Operations
- Attributes: year, month, day
- Comparisons: >, <, >=, <=, ==, !=

### Advanced Features
- Automatic null handling appropriate to data type
- Type-safe operations with proper conversions
- Multi-condition filtering with proper parentheses
- Case-sensitive and case-insensitive string operations

## Configuration
The PandasQuery constructor accepts the following parameters:

```python
PandasQuery(
    model: str = "gpt-4",              # OpenAI model to use
    temperature: float = 0.2,          # Temperature for query generation
    api_key: Optional[str] = None,     # OpenAI API key
    validate: bool = True              # Enable/disable query validation
)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## License
MIT