# src/pandas_validator.py
import pandas as pd
from typing import Dict, List, Optional


class PandasQueryValidator:
    """Validates pandas query operations and provides suggestions for corrections."""

    def __init__(self, df: pd.DataFrame):
        """Initialize validator with DataFrame schema information."""
        self.dtypes = df.dtypes.to_dict()
        self.columns = set(df.columns)

        # Valid pandas operations by data type - simplified to most common operations
        self.valid_operations = {
            'object': {
                'string_ops': {
                    'contains', 'startswith', 'endswith', 'count',  # Added count explicitly
                    'lower', 'upper', 'strip', 'len', 'slice', 'extract',
                    'find', 'findall', 'replace', 'pad', 'center', 'split'
                },
                'comparisons': {'==', '!=', 'isin'}
            },
            'number': {
                'numeric_ops': {'sum', 'mean', 'min', 'max'},
                'comparisons': {'>', '<', '>=', '<=', '==', '!='}
            },
            'datetime': {
                'date_ops': {'year', 'month', 'day'},
                'comparisons': {'>', '<', '>=', '<=', '==', '!='}
            },
            'bool': {
                'comparisons': {'==', '!='}
            }
        }

        # Common pandas aggregation functions that require groupby
        # Removed 'count' from here since it's also a string operation
        self.group_required_aggs = {
            'sum', 'mean', 'median', 'min', 'max'
        }



    def _extract_operations(self, code: str) -> Dict[str, List[str]]:
        """Extract pandas operations from the code, categorizing them by type."""
        import re

        # Match string operations specifically
        str_pattern = r'\.str\.(\w+)'
        str_ops = re.findall(str_pattern, code)

        # Match other operations, excluding string operations
        other_pattern = r'(?<!\.str)\.(\w+)\('
        other_ops = re.findall(other_pattern, code)

        return {
            'string_ops': str_ops,
            'other_ops': other_ops
        }

    def _check_aggregation_usage(self, code: str) -> list[str]:
        """Check for valid aggregation function usage."""
        errors = []
        operations = self._extract_operations(code)

        # Check only non-string operations for aggregation requirements
        for op in operations['other_ops']:
            if op in self.group_required_aggs:  # count is no longer here
                if 'groupby' not in code:
                    error_msg = f"Aggregation '{op}' used without groupby"
                    errors.append(error_msg)

        return errors

    def _check_operation_compatibility(self, code: str) -> list[str]:
        """Check if operations are compatible with column data types."""
        errors = []
        operations = self._extract_operations(code)
        column_refs = self._extract_column_references(code)

        for col in column_refs:
            if col not in self.columns:
                continue

            dtype = self.dtypes[col]
            dtype_category = (
                'number' if pd.api.types.is_numeric_dtype(dtype)
                else 'datetime' if pd.api.types.is_datetime64_dtype(dtype)
                else 'bool' if pd.api.types.is_bool_dtype(dtype)
                else 'object'
            )

            # Check string operations
            if operations['string_ops']:
                if dtype_category != 'object':
                    errors.append(
                        f"String operations used on non-string column '{col}' "
                        f"of type {dtype}"
                    )
                else:
                    for op in operations['string_ops']:
                        if op not in self.valid_operations['object']['string_ops']:
                            errors.append(
                                f"String operation '{op}' may not be valid for "
                                f"column '{col}'"
                            )

            # Check other operations
            if dtype_category in self.valid_operations:
                valid_ops = set().union(
                    *self.valid_operations[dtype_category].values()
                )
                for op in operations['other_ops']:
                    if op not in valid_ops and op not in self.group_required_aggs:
                        errors.append(
                            f"Operation '{op}' may not be compatible with "
                            f"column '{col}' of type {dtype}"
                        )

        return errors

    def _extract_column_references(self, code: str) -> set[str]:
        """Extract column references from the code."""
        import re
        # Match patterns like df['column'] or df.column
        pattern = r"df[\['](\w+)[\]']|df\.(\w+)"
        matches = re.findall(pattern, code)
        # Flatten and filter matches
        return {match[0] or match[1] for match in matches}


    def _check_column_existence(self, code: str) -> list[str]:
        """Check if all referenced columns exist in the DataFrame."""
        errors = []
        referenced_columns = self._extract_column_references(code)

        for col in referenced_columns:
            if col not in self.columns:
                similar_cols = [
                    existing_col for existing_col in self.columns
                    if existing_col.lower() == col.lower()
                ]
                error_msg = f"Column '{col}' does not exist in DataFrame"
                if similar_cols:
                    error_msg += f". Did you mean '{similar_cols[0]}'?"
                errors.append(error_msg)

        return errors


    def suggest_corrections(self, code: str) -> Optional[str]:
        """Attempt to suggest corrections for common issues."""
        corrected = code

        # Fix column name case sensitivity
        for col in self._extract_column_references(code):
            if col not in self.columns:
                for actual_col in self.columns:
                    if col.lower() == actual_col.lower():
                        corrected = corrected.replace(
                            f"['{col}']", f"['{actual_col}']"
                        )
                        corrected = corrected.replace(
                            f".{col}", f".{actual_col}"
                        )

        # Add null handling for string operations
        if '.str.' in corrected and 'fillna' not in corrected:
            corrected = corrected.replace('.str.', '.fillna("").str.')

        if corrected != code:
            return corrected
        return None

    def validate_query(self, code: str) -> tuple[bool, list[str]]:
        """Validate a pandas query code."""
        errors = []

        # Run all essential checks
        errors.extend(self._check_column_existence(code))
        errors.extend(self._check_operation_compatibility(code))
        errors.extend(self._check_aggregation_usage(code))

        return len(errors) == 0, errors

    def get_validation_result(self, code: str) -> Dict:
        """Get comprehensive validation results."""
        is_valid, errors = self.validate_query(code)
        suggested_correction = None

        if not is_valid:
            suggested_correction = self.suggest_corrections(code)

        return {
            'code': code.strip(),
            'is_valid': is_valid,
            'errors': errors,
            'suggested_correction': suggested_correction
        }