from openai import OpenAI
import pandas as pd
import numpy as np
from typing import Any, Optional, Dict
import os
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence
from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem, default_guarded_getiter
from .pandas_validator import PandasQueryValidator


class PandasQuery:
    """A streamlined class for executing natural language queries on pandas DataFrames using OpenAI's LLM."""

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
        self.last_code = None
        self.validate = validate

        # Set up sandbox environment
        self.restricted_globals = {
            "__builtins__": dict(safe_builtins),
            "pd": pd,
            "np": np,
            "_getattr_": default_guarded_getattr,
            "_getitem_": default_guarded_getitem,
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        }

        # Core pandas Series methods (excluding string accessor methods)
        self.series_methods = [
            "sum", "mean", "any", "argmax", "argmin", "count", "cumsum",
            "diff", "dropna", "fillna", "head", "idxmax", "idxmin",
            "max", "min", "notna", "prod", "quantile", "rename", "round",
            "tail", "to_frame", "to_list", "to_numpy", "unique",
            "sort_index", "sort_values", "aggregate", "isna", "astype"
        ]

        # Add series methods to restricted globals
        self.restricted_globals.update({
            method: getattr(pd.Series, method) for method in self.series_methods
        })

    def _build_prompt(self, df: pd.DataFrame, query: str, n: int = 5) -> str:
        """Build a detailed prompt with DataFrame information and query context."""
        # Get detailed column information
        column_info = []
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            unique_count = df[col].nunique()

            # Get appropriate sample values and range info
            sample_vals = df[col].sample(min(n, df[col].count()))
            if pd.api.types.is_numeric_dtype(dtype):
                try:
                    range_info = f"Range: {df[col].min()} to {df[col].max()}"
                except:
                    range_info = f"Sample values: {list(sample_vals)}"
            else:
                range_info = f"Sample values: {list(sample_vals)}"

            column_info.append(
                f"- {col} ({dtype}):\n"
                f"  * {range_info}\n"
                f"  * Null values: {null_count}\n"
                f"  * Unique values: {unique_count}"
            )

        prompt = f"""Given a pandas DataFrame with {len(df)} rows and the following columns:

{chr(10).join(column_info)}

Write a single line of Python code that answers this question: 

{query}

Guidelines:
1. Basic Requirements:
   - Use only pandas and numpy operations
   - Assign the result to a variable named 'result'
   - Return only the code, no explanations

2. Type-Specific Operations:
   - For numeric operations on string numbers: Use pd.to_numeric(df['column'], errors='coerce')
   - For string comparisons: Use .fillna('').str.lower()
   - For string pattern matching: Use .str.contains() or .str.startswith()
   - For datetime comparisons: Use .dt accessor

3. Null Handling:
   - Always handle null values before operations
   - Use fillna() for string operations
   - Use dropna() or fillna() for numeric operations

4. Available Methods:
   Core methods: {', '.join(self.series_methods)}
   String operations available via .str accessor
   DateTime operations available via .dt accessor

Example patterns:
- String to number comparison: result = df[pd.to_numeric(df['column'], errors='coerce') > 5]
- Case-insensitive search: result = df[df['column'].fillna('').str.lower().str.contains('pattern')]
- Section number filtering: result = df[df['section_number'].fillna('').str.startswith('6')]
"""
        return prompt

    def _execute_in_sandbox(self, code: str, df: pd.DataFrame) -> Any:
        """Execute code in RestrictedPython sandbox with validation."""
        try:
            # Pre-execution validation
            if self.validate:
                validator = PandasQueryValidator(df)
                validation_result = validator.validate_pandas_query(code)
                if not validation_result['is_valid']:
                    for error in validation_result['errors']:
                        print(f"Warning: {error}")
                    if validation_result['suggested_correction']:
                        print("Using suggested correction")
                        code = validation_result['suggested_correction']

            # Compile the code in restricted mode
            byte_code = compile_restricted(
                source=code,
                filename='<inline>',
                mode='exec'
            )

            # Create local namespace with DataFrame and numeric conversion function
            local_vars = {
                'df': df,
                'result': None,
                'pd': pd
            }

            # Execute in sandbox
            exec(byte_code, self.restricted_globals, local_vars)

            result = local_vars['result']
            if result is None:
                raise ValueError("Execution produced no result")

            return result

        except Exception as e:
            error_msg = f"Sandbox execution failed. Code: {code}. Error: {str(e)}"
            raise RuntimeError(error_msg)

    def execute(self, df: pd.DataFrame, query: str) -> Any:
        """Execute a natural language query with validation."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": self._build_prompt(df, query)}
            ]
        )

        code = response.choices[0].message.content.strip()

        # Clean up code
        if code.startswith("```"):
            code = code.split("\n", 1)[1].rsplit("\n", 1)[0]
        if code.startswith("python"):
            code = code.split("\n", 1)[1]
        code = code.strip("` \n")

        self.last_code = code

        # Execute in sandbox
        return self._execute_in_sandbox(code, df)