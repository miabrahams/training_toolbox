"""
Schema-driven ComfyUI workflow parser (v2 - Type-based)

This module validates and extracts fields from a ComfyUI prompt graph according to
a YAML schema that specifies field types and extraction rules.

Schema Format:
  roles:
    <role_name>:
      node_id: <node_id_in_graph>
      node_type: <expected_class_type>  # optional
      inputs:
        <field_name>: <type>  # e.g., "cfg: float", "seed: int"
      optional_group: <group_name>  # optional

  outputs:
    <output_field>: [<role>, <field_name>]

Type System:
  - text: String extraction
  - int: Integer extraction
  - float: Float extraction
  - bool: Boolean extraction
  - _enabled: Special type that returns True if the role's node exists

Optional Groups:
  - If any role in a group exists, the group is considered present
  - Outputs for roles in absent groups are omitted
  - Outputs for roles in present groups are required

Public API:
  extract_from_prompt(prompt_graph: dict, schema_path: str | Path) -> ExtractedPrompt
  extract_from_file(filename: Path, schema_path: str | Path) -> ExtractedPrompt
  extract_latest_from_prompt(prompt_graph: dict) -> ExtractedPrompt
  extract_latest_from_file(filename: Path) -> ExtractedPrompt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set, Tuple

import yaml
from pydantic import ValidationError

from src.lib.metadata import read_comfy_metadata
from src.lib.prompt_parser import clean_prompt
from src.schemas.prompt import ExtractedPrompt
from .errors import (
    NodeNotFoundError,
    NodeTypeMismatchError,
    MissingInputError,
    # EmptyValueError,
    LinkedInputError,
)
from .type_extractors import get_extractor


@dataclass(frozen=True)
class RoleSpec:
    """Describes one logical role (node) in the schema.

    Attributes:
        node_id: The node key in the prompt graph (stringified).
        node_type: Expected class_type for the node, or None to skip type check.
        inputs: Mapping from field name to type name (e.g., {"cfg": "float", "seed": "int"}).
        optional_group: Group name used to bundle optional roles.
    """

    node_id: str
    node_type: Optional[str]
    inputs: Mapping[str, str]  # field_name -> type_name
    optional_group: Optional[str] = None


@dataclass(frozen=True)
class SchemaSpec:
    """In-memory representation of the schema."""

    version: str
    name: str
    roles: Dict[str, RoleSpec]
    outputs: Dict[str, Tuple[str, str]]  # output_name -> (role, field_name)
    groups: Dict[str, Set[str]]  # optional group -> set(role names)


def _as_str(value: Any) -> str:
    """Convert value to string, handling None."""
    return str(value) if value is not None else ""


def _load_schema(schema_path: str | Path) -> SchemaSpec:
    """Load and validate the YAML schema from disk.

    The schema format uses field_name -> type_name mappings in the inputs section.
    For example:
        inputs:
          cfg: float
          seed: int
          text: text
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    roles: Dict[str, RoleSpec] = {}
    groups: Dict[str, Set[str]] = {}

    for role_name, role_def in (data.get("roles", {}) or {}).items():
        node_id_val = role_def.get("node_id")
        if node_id_val is None:
            raise ValueError(f"Schema missing node_id for role '{role_name}'")
        node_id = _as_str(node_id_val)

        node_type = role_def.get("node_type")
        inputs = role_def.get("inputs", {}) or {}
        optional_group = role_def.get("optional_group")

        roles[role_name] = RoleSpec(
            node_id=node_id,
            node_type=node_type,
            inputs=inputs,
            optional_group=optional_group,
        )
        if optional_group:
            groups.setdefault(optional_group, set()).add(role_name)

    outputs: Dict[str, Tuple[str, str]] = {}
    for out_name, spec in (data.get("outputs", {}) or {}).items():
        if isinstance(spec, list) and len(spec) == 2:
            role, field_name = spec
        elif isinstance(spec, dict):
            role = spec.get("role")
            field_name = spec.get("field") or spec.get("input")
        else:
            raise ValueError(f"Invalid output mapping for '{out_name}': {spec}")

        if not role or not field_name:
            raise ValueError(f"Output '{out_name}' missing role or field definition")

        outputs[out_name] = (str(role), str(field_name))

    return SchemaSpec(
        version=_as_str(data.get("version", "")),
        name=_as_str(data.get("name", "")),
        roles=roles,
        outputs=outputs,
        groups=groups,
    )


def _validate_outputs(schema: SchemaSpec) -> None:
    """Ensure schema outputs map to known ExtractedPrompt fields."""
    allowed = set(ExtractedPrompt.model_fields)
    unknown = sorted(k for k in schema.outputs if k not in allowed)
    if unknown:
        raise ValueError(
            "Schema outputs reference unknown ExtractedPrompt field(s): "
            + ", ".join(unknown)
        )


def _validate_role_node(role: str, spec: RoleSpec, node: Dict[str, Any]) -> None:
    """Validate a single node against the role spec (type check only)."""
    if spec.node_type:
        class_type = node.get("class_type")
        if class_type != spec.node_type:
            raise NodeTypeMismatchError(
                role=role,
                node_id=spec.node_id,
                expected_type=spec.node_type,
                actual_type=class_type or "UNKNOWN"
            )


def _determine_group_presence(
    prompt_graph: Mapping[str, Any], schema: SchemaSpec
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bool]]:
    """Resolve nodes for roles and detect which optional groups are present.

    Returns:
        resolved_nodes: role -> node for all validated roles (present or required)
        group_present: group -> bool indicating if at least one role in the group exists
    """
    resolved_nodes: Dict[str, Dict[str, Any]] = {}
    group_present: Dict[str, bool] = {g: False for g in schema.groups}

    for role_name, spec in schema.roles.items():
        node = prompt_graph.get(spec.node_id)
        if node is None:
            if spec.optional_group:
                continue
            raise NodeNotFoundError(role=role_name, node_id=spec.node_id)

        _validate_role_node(role_name, spec, node)
        resolved_nodes[role_name] = node
        if spec.optional_group:
            group_present[spec.optional_group] = True

    return resolved_nodes, group_present


def _extract_value(
    *,
    role: str,
    spec: RoleSpec,
    node: Dict[str, Any],
    field_name: str,
    output_name: str,
) -> Any:
    """Extract and transform a value from a node's inputs using type extractors.

    Args:
        role: The role name
        spec: The role specification
        node: The ComfyUI node dictionary
        field_name: The field name to extract
        output_name: The output field name (for error messages)

    Returns:
        The extracted and transformed value

    Raises:
        MissingInputError: If the field is not in the node's inputs
        LinkedInputError: If the field is a link to another node
        ValueError: If extraction/transformation fails
    """
    inputs = node.get("inputs", {}) or {}

    # Check if field exists
    if field_name not in inputs:
        available_inputs = list(inputs.keys())
        raise MissingInputError(
            role=role,
            node_id=spec.node_id,
            input_field=field_name,
            output_field=output_name,
            available_inputs=available_inputs
        )

    # Get raw value
    value = inputs[field_name]

    # Check if it's a link to another node
    if isinstance(value, list):
        linked_node_id = value[0] if value else None
        raise LinkedInputError(
            role=role,
            input_field=field_name,
            output_field=output_name,
            linked_node_id=str(linked_node_id) if linked_node_id is not None else None
        )

    # Get the type for this field
    type_name = spec.inputs.get(field_name)
    if type_name is None:
        raise ValueError(
            f"Field '{field_name}' not defined in role '{role}' inputs. "
            f"Available fields: {list(spec.inputs.keys())}"
        )

    # Apply type extraction
    try:
        extractor = get_extractor(type_name)
        return extractor.extract(value)
    except ValueError as e:
        # Wrap extraction errors with context
        raise ValueError(
            f"Failed to extract '{output_name}' from role '{role}' field '{field_name}' "
            f"as type '{type_name}': {e}"
        ) from e


def extract_from_prompt(prompt_graph: Dict[str, Any], schema_path: str | Path) -> ExtractedPrompt:
    """Validate the prompt graph against the schema and produce an ExtractedPrompt.

    Args:
        prompt_graph: The ComfyUI prompt dictionary (from image metadata)
        schema_path: Path to the YAML schema file

    Returns:
        ExtractedPrompt instance with all extracted fields

    Raises:
        NodeNotFoundError: If a required node is missing
        NodeTypeMismatchError: If a node has the wrong type
        MissingInputError: If a required input field is missing
        ValueError: For other validation errors
    """
    schema = _load_schema(schema_path)
    _validate_outputs(schema)

    # Resolve roles and detect optional group presence
    resolved_nodes, group_present = _determine_group_presence(prompt_graph, schema)

    result: Dict[str, Any] = {
        "schema_version": schema.version,
        "schema_name": schema.name,
    }

    for out_name, (role, field_name) in schema.outputs.items():
        spec = schema.roles.get(role)
        if spec is None:
            raise ValueError(f"Output '{out_name}' references unknown role '{role}'")

        # Skip outputs for roles in optional groups that are entirely absent
        if spec.optional_group and not group_present.get(spec.optional_group, False):
            continue

        # Special handling for _enabled pseudo-type
        if field_name == "_enabled":
            if role in resolved_nodes:
                result[out_name] = True
            # else: omit when not present (optional group absence handled above)
            continue

        # For regular outputs, the role must have been resolved at this point
        node = resolved_nodes.get(role)
        if node is None:
            group_msg = f" in optional group '{spec.optional_group}'" if spec.optional_group else ""
            raise ValueError(
                f"Required role '{role}'{group_msg} is missing for output '{out_name}'"
            )

        value = _extract_value(
            role=role,
            spec=spec,
            node=node,
            field_name=field_name,
            output_name=out_name,
        )
        result[out_name] = value

    # Clean the positive prompt if present
    if "positive_prompt" not in result:
        raise ValueError(
            "Schema outputs missing required field 'positive_prompt' for cleaning"
        )

    result["cleaned_prompt"] = clean_prompt(result["positive_prompt"])

    # Validate with Pydantic
    try:
        return ExtractedPrompt(**result)
    except ValidationError as ve:
        raise ValueError(
            "Extracted values do not conform to ExtractedPrompt: " + str(ve)
        ) from ve


def extract_from_file(filename: Path, schema_path: str | Path) -> ExtractedPrompt:
    """Extract from an image file using a specific schema."""
    prompt, _ = read_comfy_metadata(filename)
    return extract_from_prompt(prompt, schema_path)


# Convenience default schema path (latest located under schema_definitions/)
DEFAULT_SCHEMA_PATH = (
    Path(__file__).with_name("schema_definitions").joinpath("schema_v5.yml")
)


def extract_latest_from_prompt(prompt_graph: Dict[str, Any]) -> ExtractedPrompt:
    """Extract using the latest default schema."""
    return extract_from_prompt(prompt_graph, DEFAULT_SCHEMA_PATH)


def extract_latest_from_file(filename: Path) -> ExtractedPrompt:
    """Extract from an image file using the latest default schema."""
    return extract_from_file(filename, DEFAULT_SCHEMA_PATH)


__all__ = [
    "extract_from_prompt",
    "extract_from_file",
    "extract_latest_from_prompt",
    "extract_latest_from_file",
    "DEFAULT_SCHEMA_PATH",
    "RoleSpec",
    "SchemaSpec",
]
