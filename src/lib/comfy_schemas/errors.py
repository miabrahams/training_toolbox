"""
Enhanced error classes for ComfyUI schema extraction with detailed context.
"""
from typing import Any, Dict, Optional


class SchemaExtractionError(ValueError):
    """Base class for schema extraction errors with detailed context."""

    def __init__(
        self,
        message: str,
        *,
        role: Optional[str] = None,
        node_id: Optional[str] = None,
        node_type: Optional[str] = None,
        expected_type: Optional[str] = None,
        input_field: Optional[str] = None,
        output_field: Optional[str] = None,
        value: Optional[Any] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize with detailed context about the failure.

        Args:
            message: Human-readable error message
            role: The schema role that failed
            node_id: The ComfyUI node ID
            node_type: The actual node class_type
            expected_type: The expected node class_type
            input_field: The input field name that failed
            output_field: The output field name being extracted
            value: The problematic value
            suggestion: Suggestion for fixing the issue
        """
        super().__init__(message)
        self.role = role
        self.node_id = node_id
        self.node_type = node_type
        self.expected_type = expected_type
        self.input_field = input_field
        self.output_field = output_field
        self.value = value
        self.suggestion = suggestion

    def get_context(self) -> Dict[str, Any]:
        """Get all available context as a dictionary."""
        context = {}
        if self.role:
            context["role"] = self.role
        if self.node_id:
            context["node_id"] = self.node_id
        if self.node_type:
            context["node_type"] = self.node_type
        if self.expected_type:
            context["expected_type"] = self.expected_type
        if self.input_field:
            context["input_field"] = self.input_field
        if self.output_field:
            context["output_field"] = self.output_field
        if self.value is not None:
            context["value"] = self.value
        if self.suggestion:
            context["suggestion"] = self.suggestion
        return context


class NodeNotFoundError(SchemaExtractionError):
    """Raised when a required node is not found in the prompt graph."""

    def __init__(self, role: str, node_id: str, *, suggestion: Optional[str] = None):
        message = f"Node for role '{role}' (id={node_id}) not found in prompt graph"
        if not suggestion:
            suggestion = (
                "1. Verify the node_id in your schema matches the actual workflow\n"
                "2. If this is optional, add the role to an 'optional_group'\n"
                "3. Use --nodes to see all available node IDs"
            )
        super().__init__(
            message,
            role=role,
            node_id=node_id,
            suggestion=suggestion
        )


class NodeTypeMismatchError(SchemaExtractionError):
    """Raised when a node's class_type doesn't match the expected type."""

    def __init__(
        self,
        role: str,
        node_id: str,
        expected_type: str,
        actual_type: str,
        *,
        suggestion: Optional[str] = None
    ):
        message = (
            f"Node type mismatch for role '{role}' (id={node_id}): "
            f"expected '{expected_type}', got '{actual_type}'"
        )
        if not suggestion:
            suggestion = (
                f"Update the schema's node_type for role '{role}' from "
                f"'{expected_type}' to '{actual_type}'"
            )
        super().__init__(
            message,
            role=role,
            node_id=node_id,
            node_type=actual_type,
            expected_type=expected_type,
            suggestion=suggestion
        )


class MissingInputError(SchemaExtractionError):
    """Raised when a node is missing an expected input field."""

    def __init__(
        self,
        role: str,
        node_id: str,
        input_field: str,
        output_field: str,
        available_inputs: Optional[list] = None,
        *,
        suggestion: Optional[str] = None
    ):
        message = (
            f"Missing input '{input_field}' on node role '{role}' (id={node_id}) "
            f"for output '{output_field}'"
        )
        if available_inputs:
            message += f"\nAvailable inputs: {', '.join(available_inputs)}"

        if not suggestion:
            suggestion = (
                f"1. Check the node's actual inputs with --node {node_id}\n"
                f"2. Update the schema's 'inputs' mapping for role '{role}'\n"
            )
            if available_inputs:
                suggestion += f"3. Available inputs are: {', '.join(available_inputs)}"

        super().__init__(
            message,
            role=role,
            node_id=node_id,
            input_field=input_field,
            output_field=output_field,
            suggestion=suggestion
        )


class EmptyValueError(SchemaExtractionError):
    """Raised when an input field has an empty or null value."""

    def __init__(
        self,
        role: str,
        input_field: str,
        output_field: str,
        *,
        suggestion: Optional[str] = None
    ):
        message = f"Empty value for '{output_field}' from role '{role}' (input '{input_field}')"
        if not suggestion:
            suggestion = (
                "1. Ensure the workflow properly sets this value\n"
                "2. If this field is optional, consider making it part of an optional_group"
            )
        super().__init__(
            message,
            role=role,
            input_field=input_field,
            output_field=output_field,
            suggestion=suggestion
        )


class LinkedInputError(SchemaExtractionError):
    """Raised when an input is a link to another node instead of a literal value."""

    def __init__(
        self,
        role: str,
        input_field: str,
        output_field: str,
        linked_node_id: Optional[str] = None,
        *,
        suggestion: Optional[str] = None
    ):
        message = (
            f"Input '{input_field}' on role '{role}' is a link/reference, "
            f"expected a literal value for '{output_field}'"
        )
        if linked_node_id:
            message += f" (links to node {linked_node_id})"

        if not suggestion:
            suggestion = (
                "This input connects to another node instead of having a direct value.\n"
                "The extractor currently doesn't support traversing node links.\n"
                "Consider updating the schema to extract from the linked node instead."
            )

        super().__init__(
            message,
            role=role,
            input_field=input_field,
            output_field=output_field,
            value=linked_node_id,
            suggestion=suggestion
        )
