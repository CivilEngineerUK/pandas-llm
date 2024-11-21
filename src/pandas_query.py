from openai import OpenAI
import pandas as pd
import numpy as np
from typing import Any, Optional, Dict
import os
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence
from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem, default_guarded_getiter


class PandasQuery:
    """
    A streamlined class for executing natural language queries on pandas DataFrames using OpenAI's LLM,
    with sandbox protection.
    """

    def __init__(
            self,
            model: str = "gpt-4",
            temperature: float = 0.2,
            api_key: Optional[str] = None
    ):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.last_code = None

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

        # Add safe pandas Series methods
        series_methods = [
            "sum", "mean", "any", "argmax", "argmin", "count", "cumsum",
            "diff", "dropna", "fillna", "head", "idxmax", "idxmin",
            "max", "min", "notna", "prod", "quantile", "rename", "round",
            "tail", "to_frame", "to_list", "to_numpy", "unique",
            "sort_index", "sort_values", "aggregate"
        ]
        self.restricted_globals.update({
            method: getattr(pd.Series, method) for method in series_methods
        })

    def _build_prompt(self, df: pd.DataFrame, query: str) -> str:
        columns_info = "\n".join(f"- {col} ({df[col].dtype})" for col in df.columns)

        return f"""Given a pandas DataFrame with {len(df)} rows and the following columns:
{columns_info}

Write a single line of Python code that answers this question: {query}

Guidelines:
- Use only pandas and numpy operations
- String comparisons should be case-insensitive
- Assign the result to a variable named 'result'
- Return only the code, no explanations
"""

    def _execute_in_sandbox(self, code: str, df: pd.DataFrame) -> Any:
        """Execute code in RestrictedPython sandbox"""
        try:
            # Compile the code in restricted mode
            byte_code = compile_restricted(
                source=code,
                filename='<inline>',
                mode='exec'
            )

            # Create local namespace with just the dataframe
            local_vars = {'df': df, 'result': None}

            # Execute in sandbox
            exec(byte_code, self.restricted_globals, local_vars)

            return local_vars['result']

        except Exception as e:
            raise RuntimeError(f"Sandbox execution failed. Code: {code}. Error: {str(e)}")

    def execute(self, df: pd.DataFrame, query: str) -> Any:
        """
        Execute a natural language query on a pandas DataFrame using sandbox protection.

        Args:
            df: The pandas DataFrame to query
            query: Natural language query string

        Returns:
            Query result (could be DataFrame, Series, scalar, etc.)
        """
        # Get code from LLM
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": self._build_prompt(df, query)}
            ]
        )

        code = response.choices[0].message.content.strip()

        # Clean up code if it's in a code block
        if code.startswith("```"):
            code = code.split("\n", 1)[1].rsplit("\n", 1)[0]
        if code.startswith("python"):
            code = code.split("\n", 1)[1]
        code = code.strip("` \n")

        # Store for reference
        self.last_code = code

        # Execute in sandbox
        return self._execute_in_sandbox(code, df)