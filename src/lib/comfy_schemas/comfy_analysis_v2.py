"""
Schema-driven ComfyUI workflow parser

This module validates and extracts fields from a ComfyUI prompt graph (the JSON
stored under an image's "prompt" metadata) according to a YAML schema.

- All outputs declared in the schema are required unless their role belongs to an
  optional group that is entirely absent from the prompt graph.
- For optional groups, if any role in the group exists, the group is considered
  present; outputs referencing missing roles within a present group cause a
  validation error.
- Role type checks are enforced when a role's node_type is specified.
- Special output input key "_enabled" yields a presence flag (True) when the
  corresponding role's node is present; it is omitted otherwise.

Public API:
  extract_from_prompt(prompt_graph: dict, schema_path: str | Path) -> dict
  extract_from_file(filename: Path, schema_path: str | Path) -> dict
  extract_latest_from_prompt(prompt_graph: dict) -> dict
  extract_latest_from_file(filename: Path) -> dict

Notes:
- The default schema points to the latest bundled schema file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set, Tuple

import yaml

from src.lib.metadata import read_comfy_metadata


# -------- Data structures ---------

@dataclass(frozen=True)
class RoleSpec:
    """Describes one logical role (node) in the schema.

    Attributes:
        node_id: The node key in the prompt graph (stringified).
        node_type: Expected class_type for the node, or None to skip type check.
        inputs: Mapping from logical input keys to actual field names in node["inputs"].
        optional_group: Group name used to bundle optional roles. If any role in a
                        group is present, the group is considered present.
    """

    node_id: str
    node_type: Optional[str]
    inputs: Mapping[str, str]
    optional_group: Optional[str] = None


@dataclass(frozen=True)
class SchemaSpec:
    """Holds the parsed YAML schema in a convenient, typed form."""

    version: str
    name: str
    roles: Dict[str, RoleSpec]
    outputs: Dict[str, Tuple[str, str]]  # output_name -> (role, input_key)
    groups: Dict[str, Set[str]]  # optional group -> set(role names)


# -------- Schema loading ---------

def _as_str(value: Any) -> str:
    return str(value) if value is not None else ""


def _load_schema(schema_path: str | Path) -> SchemaSpec:
    """Load and validate the YAML schema from disk.

    The loader normalizes:
    - node_id to string (Comfy prompt dict uses string keys)
    - outputs into 2-tuples (role, input_key)
    - optional groups to a mapping of group -> roles
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
            role, input_key = spec
        elif isinstance(spec, dict):
            role = spec.get("role")
            input_key = spec.get("input")
        else:
            raise ValueError(f"Invalid output mapping for '{out_name}': {spec}")

        if not role or not input_key:
            raise ValueError(f"Output '{out_name}' missing role or input definition")

        outputs[out_name] = (str(role), str(input_key))

    return SchemaSpec(
        version=_as_str(data.get("version", "")),
        name=_as_str(data.get("name", "")),
        roles=roles,
        outputs=outputs,
        groups=groups,
    )


# -------- Validation helpers ---------

def _validate_role_node(role: str, spec: RoleSpec, node: Dict[str, Any]) -> None:
    """Validate a single node against the role spec (type check only)."""
    if spec.node_type:
        class_type = node.get("class_type")
        if class_type != spec.node_type:
            raise ValueError(
                f"Node type mismatch for role '{role}': expected '{spec.node_type}', got '{class_type}'"
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
            raise ValueError(
                f"Node for role '{role_name}' (id={spec.node_id}) not found in prompt graph"
            )

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
    logical_key: str,
    output_name: str,
) -> Any:
    """Extract and validate a value from a node's inputs for a given logical key."""
    inputs = node.get("inputs", {}) or {}
    field = spec.inputs.get(logical_key, logical_key)

    if field not in inputs:
        raise ValueError(
            f"Missing input '{field}' on node role '{role}' (id={spec.node_id}) for output '{output_name}'"
        )

    value = inputs.get(field)
    if isinstance(value, list):
        # Comfy links appear as lists; schema expects literal values for these outputs
        raise ValueError(
            f"Input '{field}' on role '{role}' is a link/reference, expected a literal value for '{output_name}'"
        )

    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(
            f"Empty value for '{output_name}' from role '{role}' (input '{field}')"
        )

    return value


# -------- Public API ---------

def extract_from_prompt(prompt_graph: Dict[str, Any], schema_path: str | Path) -> Dict[str, Any]:
    """Validate the prompt graph against the schema and extract declared outputs.

    Behavior matches comfy_analysis_v2, with clearer structure and error messages.
    """
    schema = _load_schema(schema_path)

    # Resolve roles and detect optional group presence
    resolved_nodes, group_present = _determine_group_presence(prompt_graph, schema)

    result: Dict[str, Any] = {
        "schema_version": schema.version,
        "schema_name": schema.name,
    }

    for out_name, (role, logical_key) in schema.outputs.items():
        spec = schema.roles.get(role)
        if spec is None:
            raise ValueError(f"Output '{out_name}' references unknown role '{role}'")

        # Skip outputs for roles in optional groups that are entirely absent
        if spec.optional_group and not group_present.get(spec.optional_group, False):
            continue

        # Presence flag shortcut
        if logical_key == "_enabled":
            if role in resolved_nodes:
                result[out_name] = True
            # else: omit when not present (optional group absence handled above)
            continue

        # For non-presence outputs, the role must have been resolved at this point
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
            logical_key=logical_key,
            output_name=out_name,
        )
        result[out_name] = value

    return result


def extract_from_file(filename: Path, schema_path: str | Path) -> Dict[str, Any]:
    prompt, _ = read_comfy_metadata(filename)
    return extract_from_prompt(prompt, schema_path)


# Convenience default schema path (latest located under schemas/)
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("schemas").joinpath("schema_v5.yml")


def extract_latest_from_prompt(prompt_graph: Dict[str, Any]) -> Dict[str, Any]:
    return extract_from_prompt(prompt_graph, DEFAULT_SCHEMA_PATH)


def extract_latest_from_file(filename: Path) -> Dict[str, Any]:
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