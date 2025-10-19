"""
Schema-driven ComfyUI workflow parser (v2)

- Loads a YAML schema describing roles (nodes) and outputs (fields to extract)
- Validates node ids, class types, and required input fields
- Extracts values from a prompt graph dict (the JSON stored under the Comfy image metadata 'prompt')

Public API:
  extract_from_prompt(prompt_graph: dict, schema_path: str) -> dict
  extract_from_file(filename: str, schema_path: str) -> dict

Behavior:
  - All outputs listed in YAML are required; if any node or input is missing, raises ValueError
  - Only supports the most recent schema for now (schema_v3.yml by default)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import yaml

from lib.metadata import read_comfy_metadata


@dataclass(frozen=True)
class RoleSpec:
    node_id: str
    node_type: Optional[str]
    inputs: Dict[str, str]
    optional_group: Optional[str] = None


@dataclass(frozen=True)
class SchemaSpec:
    version: str
    name: str
    roles: Dict[str, RoleSpec]
    outputs: Dict[str, Tuple[str, str]]  # output_name -> (role, input_key)
    groups: Dict[str, set]  # optional groups -> set of role names


def _load_schema(schema_path: str | Path) -> SchemaSpec:
    with open(schema_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Normalize node_id to strings to match Comfy prompt dict keys
    roles: Dict[str, RoleSpec] = {}
    groups: Dict[str, set] = {}
    for role_name, role in data.get("roles", {}).items():
        node_id_val = role.get("node_id")
        node_id = str(node_id_val) if node_id_val is not None else None
        node_type = role.get("node_type")  # optional; when omitted, skip type check
        inputs = role.get("inputs", {}) or {}
        optional_group = role.get("optional_group")
        if node_id is None:
            raise ValueError(f"Schema missing node_id for role '{role_name}'")
        roles[role_name] = RoleSpec(
            node_id=node_id, node_type=node_type, inputs=inputs, optional_group=optional_group
        )
        if optional_group:
            groups.setdefault(optional_group, set()).add(role_name)

    # Normalize outputs into (role, input) tuples
    outputs: Dict[str, Tuple[str, str]] = {}
    for out_name, spec in data.get("outputs", {}).items():
        if isinstance(spec, list) and len(spec) == 2:
            role, input_key = spec
        elif isinstance(spec, dict):
            role = spec.get("role")
            input_key = spec.get("input")
        else:
            raise ValueError(f"Invalid output mapping for '{out_name}': {spec}")
        if not role or not input_key:
            raise ValueError(f"Output '{out_name}' missing role or input definition")
        outputs[out_name] = (role, input_key)

    return SchemaSpec(
        version=str(data.get("version", "")),
        name=str(data.get("name", "")),
        roles=roles,
        outputs=outputs,
        groups=groups,
    )


def _validate_node(prompt_graph: Dict[str, Any], role: str, role_spec: RoleSpec) -> Dict[str, Any]:
    node = prompt_graph.get(role_spec.node_id)
    if node is None:
        raise ValueError(f"Node for role '{role}' (id={role_spec.node_id}) not found in prompt graph")
    if role_spec.node_type:
        class_type = node.get("class_type")
        if class_type != role_spec.node_type:
            raise ValueError(
                f"Node type mismatch for role '{role}': expected '{role_spec.node_type}', got '{class_type}'"
            )
    return node


def extract_from_prompt(prompt_graph: Dict[str, Any], schema_path: str | Path) -> Dict[str, Any]:
    schema = _load_schema(schema_path)

    # Validate all roles exist and types match
    # First pass: detect which optional groups are present (at least one node found)
    group_present: Dict[str, bool] = {g: False for g in schema.groups}
    resolved_nodes: Dict[str, Dict[str, Any]] = {}
    for role_name, role_spec in schema.roles.items():
        node = prompt_graph.get(role_spec.node_id)
        if node is None and role_spec.optional_group:
            # defer validation; may be skipped if entire group absent
            continue
        # validate (will raise if missing and not in optional group)
        resolved = _validate_node(prompt_graph, role_name, role_spec)
        resolved_nodes[role_name] = resolved
        if role_spec.optional_group:
            group_present[role_spec.optional_group] = True

    # Extract required outputs
    result: Dict[str, Any] = {"schema_version": schema.version, "schema_name": schema.name}
    for out_name, (role, input_key) in schema.outputs.items():
        role_spec = schema.roles.get(role)
        if role_spec is None:
            raise ValueError(f"Output '{out_name}' references unknown role '{role}'")
        # If role belongs to an optional group and the group isn't present in the graph, skip
        if role_spec.optional_group and not group_present.get(role_spec.optional_group, False):
            continue
        # Special case: presence-based flag
        if input_key == "_enabled":
            # Presence-based: this specific role's node must be present (validated)
            if role in resolved_nodes:
                result[out_name] = True
            # If node not resolved (missing), and it's optional, skip silently
            # (group absence already handled above). Do not raise.
            continue

        node = resolved_nodes[role]

        inputs = node.get("inputs", {})
        # The schema role maps logical input_key -> actual field in 'inputs'
        field = role_spec.inputs.get(input_key)
        if not field:
            # Allow direct name if not found in mapping
            field = input_key

        if field not in inputs:
            raise ValueError(
                f"Missing input '{field}' on node role '{role}' (id={role_spec.node_id}) for output '{out_name}'"
            )
        value = inputs.get(field)
        # For Comfy, direct values are expected (not link lists) for these fields
        if isinstance(value, list):
            raise ValueError(
                f"Input '{field}' on role '{role}' is a link/reference, expected a literal value for '{out_name}'"
            )
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise ValueError(
                f"Empty value for '{out_name}' from role '{role}' (input '{field}')"
            )
        result[out_name] = value

    return result


def extract_from_file(filename: Path, schema_path: str | Path) -> Dict[str, Any]:
    prompt, _workflow = read_comfy_metadata(filename)
    return extract_from_prompt(prompt, schema_path)


# Convenience default schema path (latest v3 located under schemas/)
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("schemas").joinpath("schema_v3.yml")


def extract_latest_from_prompt(prompt_graph: Dict[str, Any]) -> Dict[str, Any]:
    return extract_from_prompt(prompt_graph, DEFAULT_SCHEMA_PATH)


def extract_latest_from_file(filename: Path) -> Dict[str, Any]:
    return extract_from_file(filename, DEFAULT_SCHEMA_PATH)
