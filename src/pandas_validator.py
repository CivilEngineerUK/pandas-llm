import re
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional


class PandasQueryValidator:
    """Validates pandas query operations and provides suggestions for corrections."""

    def __init__(self, df: pd.DataFrame):
        """Initialize validator with DataFrame schema information."""
        self.dtypes = df.dtypes.to_dict()
        self.columns = set(df.columns)

        # Valid pandas operations by data type - simplified to most common operations
        self.valid_operations = {
            'object': {
                'string_ops': {'contains', 'startswith', 'endswith'},
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
        self.group_required_aggs = {
            'sum', 'mean', 'median', 'min', 'max', 'count'
        }

    def _extract_column_references(self, code: str) -> set[str]:
        """Extract column references from the code."""
        import re
        # Match patterns like df['column'] or df.column
        pattern = r"df[\['](\w+)[\]']|df\.(\w+)"
        matches = re.findall(pattern, code)
        # Flatten and filter matches
        return {match[0] or match[1] for match in matches}

    def _extract_operations(self, code: str) -> list[str]:
        """Extract pandas operations from the code."""
        import re
        # Match method calls on df or column references
        pattern = r'\.(\w+)\('
        return re.findall(pattern, code)

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

            if dtype_category in self.valid_operations:
                valid_ops = set().union(
                    *self.valid_operations[dtype_category].values()
                )

                for op in operations:
                    if op not in valid_ops and op not in self.group_required_aggs:
                        error_msg = (
                            f"Operation '{op}' may not be compatible with "
                            f"column '{col}' of type {dtype}"
                        )
                        errors.append(error_msg)

        return errors

    def _check_aggregation_usage(self, code: str) -> list[str]:
        """Check for valid aggregation function usage."""
        errors = []
        operations = self._extract_operations(code)

        for op in operations:
            if op in self.group_required_aggs:
                if 'groupby' not in code and not any(
                        c in code for c in ['sum()', 'mean()', 'count()']
                ):
                    error_msg = f"Aggregation '{op}' used without groupby"
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