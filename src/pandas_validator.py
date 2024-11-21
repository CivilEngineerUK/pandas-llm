import re
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import logging




class PandasQueryValidator:
    """Validates pandas query operations and provides suggestions for corrections."""

    def __init__(self, df: pd.DataFrame, logger: logging.Logger):
        """Initialize validator with DataFrame schema information."""
        self.dtypes = df.dtypes.to_dict()
        self.columns = set(df.columns)

        # Valid pandas operations by data type
        self.valid_operations = {
            'object': {
                'string_ops': {'contains', 'startswith', 'endswith', 'lower', 'upper', 'strip', 'len'},
                'comparisons': {'==', '!=', 'isin'}
            },
            'number': {
                'numeric_ops': {'sum', 'mean', 'min', 'max', 'count', 'median'},
                'comparisons': {'>', '<', '>=', '<=', '==', '!='}
            },
            'datetime': {
                'date_ops': {'year', 'month', 'day', 'hour', 'minute'},
                'comparisons': {'>', '<', '>=', '<=', '==', '!='}
            },
            'bool': {
                'bool_ops': {'any', 'all'},
                'comparisons': {'==', '!='}
            }
        }

        # Common pandas aggregation functions
        self.valid_aggregations = {
            'sum', 'mean', 'median', 'min', 'max', 'count',
            'std', 'var', 'first', 'last'
        }

        # Valid pandas commands and their requirements
        self.valid_commands = {
            'groupby': {'columns'},
            'agg': {'groupby'},
            'sort_values': {'columns'},
            'fillna': {'value'},
            'dropna': set(),
            'reset_index': set(),
            'merge': {'right', 'on', 'how'},
            'join': {'on'},
            'head': set(),
            'tail': set()
        }
        self.logger = logger
        self.logger.info("PandasQueryValidator initialized successfully")

    def _extract_column_references(self, code: str) -> List[str]:
        """Extract column references from the code."""
        # Match patterns like df['column'] or df.column
        pattern = r"df[\['](\w+)[\]']|df\.(\w+)"
        matches = re.findall(pattern, code)
        # Flatten and filter matches
        columns = {match[0] or match[1] for match in matches}
        self.logger.debug(f"Extracted columns: {columns}")
        return list(columns)

    def _extract_operations(self, code: str) -> List[str]:
        """Extract pandas operations from the code."""
        # Match method calls on df or column references
        pattern = r'\.(\w+)\('
        operations = re.findall(pattern, code)
        self.logger.debug(f"Extracted operations: {operations}")
        return operations

    def _check_column_existence(self, code: str) -> List[str]:
        """Check if all referenced columns exist in the DataFrame."""
        errors = []
        referenced_columns = self._extract_column_references(code)

        for col in referenced_columns:
            if col not in self.columns:
                error_msg = f"Column '{col}' does not exist in DataFrame"
                errors.append(error_msg)
                self.logger.warning(error_msg)

        return errors

    def _check_operation_compatibility(self, code: str) -> List[str]:
        """Check if operations are compatible with column data types."""
        errors = []
        operations = self._extract_operations(code)
        column_refs = self._extract_column_references(code)

        for col in column_refs:
            if col not in self.columns:
                continue

            dtype = self.dtypes[col]
            dtype_category = 'number' if pd.api.types.is_numeric_dtype(dtype) else \
                'datetime' if pd.api.types.is_datetime64_dtype(dtype) else \
                    'bool' if pd.api.types.is_bool_dtype(dtype) else 'object'

            valid_ops = set()
            if dtype_category in self.valid_operations:
                for ops in self.valid_operations[dtype_category].values():
                    valid_ops.update(ops)

            for op in operations:
                if op not in valid_ops and op not in self.valid_commands:
                    error_msg = f"Operation '{op}' may not be compatible with column '{col}' of type {dtype}"
                    errors.append(error_msg)
                    self.logger.warning(error_msg)

        return errors

    def _check_null_handling(self, code: str) -> List[str]:
        """Check for proper null value handling."""
        errors = []

        # Check for string operations without null handling
        if any(op in code for op in ['.str.', '.dt.']):
            if 'fillna' not in code and 'dropna' not in code:
                error_msg = "String or datetime operations detected without null handling"
                errors.append(error_msg)
                self.logger.warning(error_msg)

        return errors

    def _check_aggregation_usage(self, code: str) -> List[str]:
        """Check for valid aggregation function usage."""
        errors = []
        operations = self._extract_operations(code)

        for op in operations:
            if op.lower() in self.valid_aggregations:
                # Check if groupby is used before aggregation
                if 'groupby' not in code and not any(c in code for c in ['sum()', 'mean()', 'count()']):
                    error_msg = f"Aggregation '{op}' used without groupby"
                    errors.append(error_msg)
                    self.logger.warning(error_msg)

        return errors

    def suggest_corrections(self, code: str) -> Optional[str]:
        """Attempt to suggest corrections for common issues."""
        self.logger.info("Attempting to suggest corrections")

        corrected = code

        # Fix column name case sensitivity
        for col in self._extract_column_references(code):
            if col not in self.columns:
                for actual_col in self.columns:
                    if col.lower() == actual_col.lower():
                        corrected = corrected.replace(f"['{col}']", f"['{actual_col}']")
                        corrected = corrected.replace(f".{col}", f".{actual_col}")
                        self.logger.info(f"Suggested correction for column name: {col} -> {actual_col}")

        # Add null handling for string operations
        if '.str.' in corrected and 'fillna' not in corrected:
            corrected = corrected.replace('.str.', '.fillna("").str.')
            self.logger.info("Added null handling for string operations")

        # Suggest proper aggregation syntax
        if any(op in corrected for op in self.valid_aggregations) and 'groupby' not in corrected:
            self.logger.info("Suggested using groupby before aggregation")

        if corrected != code:
            return corrected
        return None

    def validate_query(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate a pandas query code.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        self.logger.info("Starting pandas query validation")

        # Run all checks
        errors.extend(self._check_column_existence(code))
        errors.extend(self._check_operation_compatibility(code))
        errors.extend(self._check_null_handling(code))
        errors.extend(self._check_aggregation_usage(code))

        is_valid = len(errors) == 0
        if is_valid:
            self.logger.info("Query is valid")
        else:
            self.logger.info("Query validation failed with errors")

        return is_valid, errors


def validate_pandas_query(df: pd.DataFrame, code: str, logger: logging.Logger) -> Dict:
    """
    Validate a pandas query and suggest corrections if needed.

    Args:
        df: Input DataFrame
        code: Pandas query code to validate

    Returns:
        Dictionary containing:
        - 'code': Original code string
        - 'is_valid': Boolean indicating if code is valid
        - 'errors': List of validation errors
        - 'suggested_correction': Suggested correction string or None
    """
    validator = PandasQueryValidator(df, logger)
    is_valid, errors = validator.validate_query(code)
    suggested_correction = None

    if not is_valid:
        suggested_correction = validator.suggest_corrections(code)

    return {
        'code': code.strip(),
        'is_valid': is_valid,
        'errors': errors,
        'suggested_correction': suggested_correction
    }