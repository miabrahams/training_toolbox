"""
Type extractors for schema-based ComfyUI extraction.

Each type extractor knows how to extract and transform a value from a ComfyUI node input.
"""
from typing import Any, Protocol


class TypeExtractor(Protocol):
    """Protocol for type extractors."""

    def extract(self, value: Any) -> Any:
        """Extract and transform a value from a node input.

        Args:
            value: The raw value from the node's inputs

        Returns:
            The transformed value

        Raises:
            ValueError: If the value cannot be extracted/transformed
        """
        ...


class TextExtractor:
    """Extracts string values."""

    def extract(self, value: Any) -> str:
        """Convert value to string."""
        if value is None:
            raise ValueError("Value is None")
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("Empty string")
            return value
        return str(value)


class IntExtractor:
    """Extracts integer values."""

    def extract(self, value: Any) -> int:
        """Convert value to integer."""
        if value is None:
            raise ValueError("Value is None")
        if isinstance(value, bool):
            # Prevent bool from being converted to int (True -> 1, False -> 0)
            raise ValueError(f"Cannot convert bool to int: {value}")
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert to int: {value}") from e


class FloatExtractor:
    """Extracts float values."""

    def extract(self, value: Any) -> float:
        """Convert value to float."""
        if value is None:
            raise ValueError("Value is None")
        if isinstance(value, bool):
            # Prevent bool from being converted to float
            raise ValueError(f"Cannot convert bool to float: {value}")
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert to float: {value}") from e


class BoolExtractor:
    """Extracts boolean values."""

    def extract(self, value: Any) -> bool:
        """Convert value to boolean."""
        if value is None:
            raise ValueError("Value is None")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"true", "yes", "1", "on"}:
                return True
            if lower in {"false", "no", "0", "off"}:
                return False
            raise ValueError(f"Cannot convert string to bool: {value}")
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError(f"Cannot convert to bool: {value}")


# Registry of available type extractors
TYPE_EXTRACTORS = {
    "text": TextExtractor(),
    "int": IntExtractor(),
    "float": FloatExtractor(),
    "bool": BoolExtractor(),
}


def get_extractor(type_name: str) -> TypeExtractor:
    """Get a type extractor by name.

    Args:
        type_name: The name of the type (e.g., "text", "int", "float", "bool")

    Returns:
        The type extractor instance

    Raises:
        ValueError: If the type is not recognized
    """
    extractor = TYPE_EXTRACTORS.get(type_name)
    if extractor is None:
        available = ", ".join(sorted(TYPE_EXTRACTORS.keys()))
        raise ValueError(
            f"Unknown type '{type_name}'. Available types: {available}"
        )
    return extractor
