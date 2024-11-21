
import pandas as pd
import numpy as np
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence
from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem, default_guarded_getiter
from openai import OpenAI
import os
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, validator
from .pandas_validator import PandasQueryValidator


class QueryResult(BaseModel):
    """Pydantic model for query execution results."""
    query: str = Field(..., description="Original query string")
    code: str = Field(..., description="Generated pandas code")
    is_valid: bool = Field(..., description="Whether the query is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation/execution errors")
    result: Optional[Any] = Field(None, description="Query execution result")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.DataFrame: lambda df: df.to_dict(orient='records'),
            pd.Series: lambda s: s.to_dict(),
            np.ndarray: lambda arr: arr.tolist(),
            np.int64: lambda x: int(x),
            np.float64: lambda x: float(x)
        }

    def _serialize_value(self, v: Any) -> Any:
        """Helper method to serialize values."""
        if isinstance(v, pd.DataFrame):
            return v.to_dict(orient='records')
        elif isinstance(v, pd.Series):
            return v.to_dict()
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (np.int64, np.float64)):
            return float(v)
        elif isinstance(v, dict):
            return {k: self._serialize_value(v) for k, v in v.items()}
        elif isinstance(v, list):
            return [self._serialize_value(item) for item in v]
        return v

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to ensure all values are serializable."""
        data = {
            'query': self.query,
            'code': self.code,
            'is_valid': self.is_valid,
            'errors': self.errors,
            'result': self._serialize_value(self.result),
        }
        return data

    def get_results(self) -> Dict[str, Any]:
        """Get a simplified dictionary of just the key results."""
        return {
            'valid': self.is_valid,
            'result': self._serialize_value(self.result) if self.is_valid else None,
            'errors': self.errors if not self.is_valid else [],
        }

    @validator('result', pre=True)
    def validate_result(cls, v):
        """Convert pandas/numpy results to native Python types."""
        if isinstance(v, pd.DataFrame):
            return v.to_dict(orient='records')
        elif isinstance(v, pd.Series):
            return v.to_dict()
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (np.int64, np.float64)):
            return float(v)
        return v


class PandasQuery:
    def __init__(
            self,
            model: str = "gpt-4",
            temperature: float = 0.2,
            api_key: Optional[str] = None,
            validate: bool = True
    ):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.validate = validate
        self.restricted_globals = self._setup_restricted_globals()

    def execute(self, df: pd.DataFrame, query: str) -> QueryResult:
        """Execute a natural language query with validation and return comprehensive results."""
        import time

        # Initialize result with Pydantic model
        query_result = QueryResult(
            query=query,
            code="",
            is_valid=False,
            errors=[],
            result=None,
        )

        try:
            # Get code from LLM
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": self._build_prompt(df, query)}
                ]
            )

            code = response.choices[0].message.content.strip()
            code = self._clean_code(code)
            query_result.code = code

            # Validate if required
            if self.validate:
                validator = PandasQueryValidator(df)
                validation_result = validator.get_validation_result(code)

                if not validation_result['is_valid']:
                    query_result.errors = validation_result['errors']
                    return query_result

                # Use suggested correction if available
                if validation_result['suggested_correction']:
                    code = validation_result['suggested_correction']
                    query_result.code = code

            # Execute if valid
            result = self._execute_in_sandbox(code, df)

            query_result.is_valid = True
            query_result.result = result

        except Exception as e:
            query_result.errors.append(f"Execution error: {str(e)}")

        return query_result

    def _setup_restricted_globals(self) -> Dict:
        """Set up restricted globals for sandbox execution."""
        # Core pandas Series methods
        series_methods = [
            "sum", "mean", "any", "argmax", "argmin", "count",
            "diff", "dropna", "fillna", "head", "max", "min",
            "sort_values", "unique", "isna", "astype"
        ]

        restricted_globals = {
            "__builtins__": dict(safe_builtins),
            "pd": pd,
            "np": np,
            "_getattr_": default_guarded_getattr,
            "_getitem_": default_guarded_getitem,
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        }

        # Add series methods
        restricted_globals.update({
            method: getattr(pd.Series, method) for method in series_methods
        })

        return restricted_globals

    def _build_prompt(self, df: pd.DataFrame, query: str) -> str:
        """Build a detailed prompt with DataFrame information and query context."""
        # Convert DataFrame info to dictionary for better LLM interpretation
        df_info = {
            "metadata": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "columns": {}
        }

        for col in df.columns:
            df_info["columns"][col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().sample(min(3, len(df))).tolist()
            }
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                df_info["columns"][col].update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None
                })

        prompt = f"""Given a pandas DataFrame with the following structure:
    ```
    {df_info}
    ```

    Write a single line of Python code that answers this question: {query}

    Requirements:
    1. Assign result to 'result' variable
    2. Handle null values appropriately:
       - For string operations: Use .fillna('') before .str operations
       - For numeric operations: Use .fillna(0) or .dropna() as appropriate
       - For boolean operations: Use .fillna(False)

    3. String Operations Guidelines:
       - Use .str accessor for string operations
       - For case-insensitive matching: Use .str.lower()
       - For counting: Use .str.count(pattern)
       - For starts/ends with: Use .str.startswith() or .str.endswith()
       - For contains: Use .str.contains(pattern, case=True/False)
       - Always handle null values before string operations

    4. Numeric Operations Guidelines:
       - For string-to-numeric conversion: Use pd.to_numeric(df['column'], errors='coerce')
       - For aggregations (sum, mean, etc.), only use with groupby
       - For comparisons, use standard operators (>, <, >=, <=, ==, !=)

    5. Date Operations Guidelines:
       - Use .dt accessor for datetime operations
       - Common attributes: .dt.year, .dt.month, .dt.day
       - For date comparisons, use standard operators

    6. Filtering Guidelines:
       - Use boolean indexing: df[condition]
       - For multiple conditions, use & (and) and | (or) with parentheses
       - Example: df[(condition1) & (condition2)]

    7. Return Guidelines:
       - Return only the matching rows unless aggregation is specifically requested
       - Do not include explanatory comments in the code
       - Keep to a single line of code

    Available String Operations:
    - Basic: contains, startswith, endswith, lower, upper, strip
    - Count/Match: count, match, extract, find, findall
    - Transform: replace, pad, center, slice, split

    Available Numeric Operations:
    - Comparisons: >, <, >=, <=, ==, !=
    - Aggregations (with groupby only): sum, mean, median, min, max, count

    Example Patterns:
    - String search: df[df['column'].fillna('').str.contains('pattern')]
    - Multiple conditions: df[(df['col1'] > 0) & (df['col2'].str.startswith('prefix'))]
    - Numeric filtering: df[pd.to_numeric(df['column'], errors='coerce') > value]
    - Case-insensitive: df[df['column'].fillna('').str.lower().str.contains('pattern')]

    Return only the code, no explanations."""

        return prompt



    @staticmethod
    def _clean_code(code: str) -> str:
        """Clean up code from LLM response."""
        if code.startswith("```"):
            code = code.split("\n", 1)[1].rsplit("\n", 1)[0]
        if code.startswith("python"):
            code = code.split("\n", 1)[1]
        return code.strip("` \n")

    def _execute_in_sandbox(self, code: str, df: pd.DataFrame) -> Any:
        """Execute code in RestrictedPython sandbox."""
        byte_code = compile_restricted(
            source=code,
            filename='<inline>',
            mode='exec'
        )

        local_vars = {'df': df, 'result': None, 'pd': pd}
        exec(byte_code, self.restricted_globals, local_vars)

        if local_vars['result'] is None:
            raise ValueError("Execution produced no result")

        return local_vars['result']

    @staticmethod
    def _extract_column_references(code: str) -> set[str]:
        """Extract column references from code."""
        import re
        pattern = r"df[\['](\w+)[\]']|df\.(\w+)"
        matches = re.findall(pattern, code)
        return {match[0] or match[1] for match in matches}