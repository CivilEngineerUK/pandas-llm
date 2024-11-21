from openai import OpenAI
import pandas as pd
import numpy as np
from typing import Any, Optional, Dict
import os
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence
from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem, default_guarded_getiter
from .pandas_validator import validate_pandas_query
import logging

logger = logging.getLogger(__name__)


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

        # Add safe pandas Series methods
        self.series_methods = [
            "sum", "mean", "any", "argmax", "argmin", "count", "cumsum",
            "diff", "dropna", "fillna", "head", "idxmax", "idxmin",
            "max", "min", "notna", "prod", "quantile", "rename", "round",
            "tail", "to_frame", "to_list", "to_numpy", "unique",
            "sort_index", "sort_values", "aggregate", "isna", "fillna"
        ]
        self.restricted_globals.update({
            method: getattr(pd.Series, method) for method in self.series_methods
        })

    def _build_prompt(self, df: pd.DataFrame, query: str) -> str:
        """
        Build a detailed prompt for the LLM that includes DataFrame information
        and guidelines for generating safe, valid code.
        """
        # Get detailed column information
        column_info = []
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            unique_count = df[col].nunique()
            sample = str(df[col].iloc[:3].tolist())

            column_info.append(
                f"- {col} ({dtype}):\n"
                f"  * Null values: {null_count}\n"
                f"  * Unique values: {unique_count}\n"
                f"  * Sample values: {sample}"
            )

        column_details = "\n".join(column_info)

        # Build comprehensive prompt
        prompt = f"""Given a pandas DataFrame with {len(df)} rows and the following columns:

{column_details}

Write a single line of Python code that answers this question: {query}

Guidelines:
1. Basic Requirements:
   - Use only pandas and numpy operations
   - Assign the result to a variable named 'result'
   - Return only the code, no explanations

2. String Operations:
   - Always handle null values before string operations using fillna('')
   - Use str.lower() for case-insensitive comparisons
   - Use str.contains() instead of direct string matching when appropriate

3. Data Type Considerations:
   - Use appropriate methods for each data type
   - For dates: Use dt accessor for date components
   - For numbers: Use appropriate numeric operations
   - For strings: Use str accessor methods

4. Null Value Handling:
   - Always consider null values in your operations
   - Use fillna() or dropna() as appropriate
   - For string operations, use fillna('') before the operation

5. Available Series Methods:
   {', '.join(self.series_methods)}

Example valid patterns:
- result = df[df['text_column'].fillna('').str.lower().str.contains('pattern')]
- result = df.groupby('column')['value'].mean()
- result = df[df['number'] > df['number'].mean()]
"""
        return prompt

    def _execute_in_sandbox(self, code: str, df: pd.DataFrame) -> Any:
        """
        Execute code in RestrictedPython sandbox with comprehensive error handling
        and safety checks.
        """
        try:
            # Pre-execution validation
            if self.validate:
                validation_result = validate_pandas_query(df, code, logger)
                if not validation_result['is_valid']:
                    logger.warning("Pre-execution validation failed:")
                    for error in validation_result['errors']:
                        logger.warning(f"- {error}")
                    if validation_result['suggested_correction']:
                        logger.info("Using suggested correction")
                        code = validation_result['suggested_correction']

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

            # Post-execution type checking
            result = local_vars['result']
            if result is None:
                raise ValueError("Execution produced no result")

            # Log successful execution
            logger.info(f"Successfully executed code: {code}")
            logger.info(f"Result type: {type(result)}")

            return result

        except Exception as e:
            error_msg = f"Sandbox execution failed. Code: {code}. Error: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def execute(self, df: pd.DataFrame, query: str) -> Any:
        """
        Execute a natural language query on a pandas DataFrame using sandbox protection
        and validation.

        Args:
            df: The pandas DataFrame to query
            query: Natural language query string

        Returns:
            Query result (could be DataFrame, Series, scalar, etc.)
        """
        logger.info(f"Executing query: {query}")

        # Get code from LLM
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

        # Store for reference
        self.last_code = code
        logger.info(f"Generated code: {code}")

        # Execute in sandbox
        return self._execute_in_sandbox(code, df)