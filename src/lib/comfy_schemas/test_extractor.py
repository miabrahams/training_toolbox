#!/usr/bin/env python3
"""
CLI tool for testing ComfyUI schema extraction with detailed error reporting.

Usage:
    python -m src.lib.comfy_schemas.test_extractor <image_path> <schema_path> [options]

Examples:
    # Test with paths relative to test/data and schema_definitions
    python -m src.lib.comfy_schemas.test_extractor test_schema_v5.png schema_v5.yml

    # Test with absolute paths
    python -m src.lib.comfy_schemas.test_extractor /path/to/image.png /path/to/schema.yml

    # Dump the prompt JSON for debugging
    python -m src.lib.comfy_schemas.test_extractor test_schema_v5.png schema_v5.yml --dump-prompt

    # Show node structure for schema development
    python -m src.lib.comfy_schemas.test_extractor test_schema_v5.png --nodes-only
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from src.lib.metadata import read_comfy_metadata
from src.lib.comfy_schemas.comfy_analysis_v2 import extract_from_prompt
from src.lib.comfy_schemas.errors import SchemaExtractionError
from src.schemas.prompt import ExtractedPrompt


# Paths relative to this file
SCRIPT_DIR = Path(__file__).parent
TEST_DATA_DIR = SCRIPT_DIR / "test" / "data"
SCHEMA_DIR = SCRIPT_DIR / "schema_definitions"


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(msg: str):
    """Print a section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_success(msg: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def print_warning(msg: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


def print_info(msg: str):
    """Print an info message."""
    print(f"{Colors.OKCYAN}{msg}{Colors.ENDC}")


def resolve_path(path_str: str, base_dir: Path, file_type: str) -> Path:
    """Resolve a path, checking both absolute and relative to base_dir."""
    path = Path(path_str)

    if path.is_absolute():
        if not path.exists():
            print_error(f"{file_type} not found: {path}")
            sys.exit(1)
        return path

    # Try relative to base_dir
    relative_path = base_dir / path
    if relative_path.exists():
        return relative_path

    # Try relative to current working directory
    if path.exists():
        return path

    print_error(f"{file_type} not found. Tried:")
    print(f"  - {relative_path}")
    print(f"  - {path.absolute()}")
    sys.exit(1)


def print_node_summary(prompt_graph: Dict[str, Any]):
    """Print a summary of nodes in the prompt graph for schema development."""
    print_header("Prompt Graph Node Summary")

    if not prompt_graph:
        print_warning("Prompt graph is empty!")
        return

    print(f"Total nodes: {len(prompt_graph)}\n")

    # Group nodes by class_type
    by_type: Dict[str, list] = {}
    for node_id, node in prompt_graph.items():
        class_type = node.get("class_type", "UNKNOWN")
        by_type.setdefault(class_type, []).append(node_id)

    # Print organized by type
    for class_type in sorted(by_type.keys()):
        node_ids = by_type[class_type]
        print(f"{Colors.BOLD}{class_type}{Colors.ENDC} ({len(node_ids)} node(s))")
        for node_id in sorted(node_ids, key=lambda x: int(x) if x.isdigit() else 0):
            node = prompt_graph[node_id]
            inputs = node.get("inputs", {})

            # Show node ID and input keys
            input_keys = list(inputs.keys()) if inputs else []
            input_str = ", ".join(input_keys[:5])  # Show first 5
            if len(input_keys) > 5:
                input_str += f", ... ({len(input_keys)} total)"

            print(f"  Node {node_id}: inputs=[{input_str}]")

    print()


def print_node_details(prompt_graph: Dict[str, Any], node_id: str):
    """Print detailed information about a specific node."""
    if node_id not in prompt_graph:
        print_error(f"Node {node_id} not found in prompt graph")
        available = ", ".join(sorted(prompt_graph.keys(), key=lambda x: int(x) if x.isdigit() else 0)[:10])
        print(f"Available nodes (first 10): {available}")
        return

    node = prompt_graph[node_id]
    print_header(f"Node {node_id} Details")

    print(f"{Colors.BOLD}Class Type:{Colors.ENDC} {node.get('class_type', 'UNKNOWN')}")

    inputs = node.get("inputs", {})
    if inputs:
        print(f"\n{Colors.BOLD}Inputs:{Colors.ENDC}")
        for key, value in inputs.items():
            if isinstance(value, list):
                print(f"  {key}: {Colors.WARNING}[LINK to node {value[0]}]{Colors.ENDC}")
            elif isinstance(value, str) and len(value) > 100:
                print(f"  {key}: \"{value[:100]}...\" ({len(value)} chars)")
            else:
                print(f"  {key}: {value!r}")
    else:
        print(f"\n{Colors.WARNING}No inputs{Colors.ENDC}")

    print()


def print_extraction_result(extracted: ExtractedPrompt):
    """Print the extracted prompt in a formatted way."""
    print_header("Extraction Result")

    # Group fields by category
    metadata = {
        "Schema Version": extracted.schema_version,
        "Schema Name": extracted.schema_name,
    }

    core_fields = {
        "Positive Prompt": extracted.positive_prompt,
        "Negative Prompt": extracted.negative_prompt,
        "Cleaned Prompt": extracted.cleaned_prompt,
    }

    generation = {
        "Checkpoint": extracted.checkpoint,
        "LoRAs": extracted.loras,
        "Steps": extracted.steps,
        "CFG": extracted.cfg,
        "Sampler": extracted.sampler_name,
        "Scheduler": extracted.scheduler,
        "Seed": extracted.seed,
        "Width": extracted.width,
        "Height": extracted.height,
        "Aspect Ratio": extracted.aspect_ratio,
        "Swap Dimensions": extracted.swap_dimensions,
    }

    optional_features = {
        "IP-Adapter Enabled": extracted.ip_enabled,
        "IP Image": extracted.ip_image,
        "IP Weight": extracted.ip_weight,
        "Rescale CFG": extracted.rescale_cfg,
        "Perp Neg": extracted.perp_neg,
    }

    def print_section(title: str, fields: dict):
        print(f"{Colors.BOLD}{title}:{Colors.ENDC}")
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, str) and len(value) > 200:
                value_str = f"{value[:200]}... ({len(value)} chars)"
            else:
                value_str = str(value)
            print(f"  {key}: {Colors.OKGREEN}{value_str}{Colors.ENDC}")
        print()

    print_section("Metadata", metadata)
    print_section("Prompts", core_fields)
    print_section("Generation Parameters", generation)

    # Only show optional features if any are present
    if any(v is not None for v in optional_features.values()):
        print_section("Optional Features", optional_features)


def print_detailed_error(error: Exception, prompt_graph: Dict[str, Any], schema_path: Path):
    """Print detailed error information to help debug schema issues."""
    print_header("Extraction Failed")

    print(f"{Colors.FAIL}{Colors.BOLD}Error Type:{Colors.ENDC} {error.__class__.__name__}")
    print(f"{Colors.FAIL}{Colors.BOLD}Error Message:{Colors.ENDC} {error}\n")

    # If this is our custom error class, extract and display context
    if isinstance(error, SchemaExtractionError):
        context = error.get_context()
        if context:
            print(f"{Colors.BOLD}Error Context:{Colors.ENDC}")
            for key, value in context.items():
                if key == "suggestion":
                    continue  # Handle separately below
                print(f"  {key}: {Colors.WARNING}{value}{Colors.ENDC}")
            print()

        # Show suggestion if available
        if error.suggestion:
            print(f"{Colors.BOLD}Suggestions:{Colors.ENDC}")
            for line in error.suggestion.split('\n'):
                if line.strip():
                    print(f"  {line}")
            print()

        # Add quick action suggestions based on context
        if context.get('node_id'):
            print(f"{Colors.BOLD}Quick actions:{Colors.ENDC}")
            print(f"  View node details: --node {context['node_id']}")
            print(f"  Dump prompt JSON: --dump-prompt")
            print()

    else:
        # Fallback for non-custom errors
        error_msg = str(error).lower()

        # Provide context-specific help based on error type
        if "node for role" in error_msg and "not found" in error_msg:
            print_info("This error means a required node is missing from the prompt graph.")
            print_info("Possible fixes:")
            print("  1. Check if the node_id in the schema matches the actual node ID in the workflow")
            print("  2. If this is an optional feature, add the role to an 'optional_group'")
            print("  3. Use --dump-prompt to see all available node IDs\n")

        elif "node type mismatch" in error_msg:
            print_info("The node exists but has a different class_type than expected.")
            print_info("Possible fixes:")
            print("  1. Update the node_type in the schema to match the actual class_type")
            print("  2. Check if the workflow uses a different node implementation\n")

        elif "missing input" in error_msg:
            print_info("The node exists but doesn't have the expected input field.")
            print_info("Possible fixes:")
            print("  1. Check the node's actual inputs with --dump-prompt or --node <id>")
            print("  2. Update the input mapping in the schema's 'inputs' section\n")

        elif "empty value" in error_msg:
            print_info("The input field exists but contains an empty value.")
            print_info("Possible fixes:")
            print("  1. Check if the workflow properly sets this value")
            print("  2. Make this field optional if it's not always needed\n")

        elif "is a link/reference" in error_msg:
            print_info("The input points to another node instead of a literal value.")
            print_info("Possible fixes:")
            print("  1. The schema may need to traverse the link to get the value")
            print("  2. This feature may require enhancement to the extractor\n")

        elif "validation" in error_msg or "pydantic" in error_msg:
            print_info("The extracted values don't match the ExtractedPrompt schema.")
            print_info("Possible fixes:")
            print("  1. Check that extracted values have the correct types (int, float, bool, str)")
            print("  2. Verify that required fields are present\n")

        # Suggest next steps
        print(f"{Colors.BOLD}Suggested next steps:{Colors.ENDC}")
        print("  1. Run with --dump-prompt to see the full prompt JSON")
        print("  2. Run with --nodes to see all available nodes and their inputs")
        print("  3. Check the schema file at:", schema_path)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Test ComfyUI schema extraction with detailed error reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "image",
        help="Image file path (absolute or relative to test/data/)"
    )

    parser.add_argument(
        "schema",
        nargs="?",
        help="Schema file path (absolute or relative to schema_definitions/). If omitted with --nodes-only, schema validation is skipped."
    )

    parser.add_argument(
        "--dump-prompt",
        action="store_true",
        help="Dump the prompt JSON to a file for debugging"
    )

    parser.add_argument(
        "--dump-workflow",
        action="store_true",
        help="Dump the workflow JSON to a file for debugging"
    )

    parser.add_argument(
        "--nodes",
        action="store_true",
        help="Show a summary of all nodes in the prompt graph"
    )

    parser.add_argument(
        "--nodes-only",
        action="store_true",
        help="Only show node summary, skip extraction"
    )

    parser.add_argument(
        "--node",
        metavar="NODE_ID",
        help="Show detailed information about a specific node"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')

    # Resolve image path
    image_path = resolve_path(args.image, TEST_DATA_DIR, "Image")
    print_success(f"Found image: {image_path}")

    # Read metadata
    print_info("Reading ComfyUI metadata from image...")
    try:
        prompt, workflow = read_comfy_metadata(image_path)
    except Exception as e:
        print_error(f"Failed to read metadata: {e}")
        sys.exit(1)

    if not prompt:
        print_error("No prompt metadata found in image!")
        print_info("This image may not be a ComfyUI-generated image, or the metadata was stripped.")
        sys.exit(1)

    print_success(f"Loaded prompt with {len(prompt)} nodes")

    # Dump prompt/workflow if requested
    if args.dump_prompt:
        output_file = image_path.with_suffix('.prompt.json')
        with open(output_file, 'w') as f:
            json.dump(prompt, f, indent=2)
        print_success(f"Dumped prompt JSON to: {output_file}")

    if args.dump_workflow:
        if workflow:
            output_file = image_path.with_suffix('.workflow.json')
            with open(output_file, 'w') as f:
                json.dump(workflow, f, indent=2)
            print_success(f"Dumped workflow JSON to: {output_file}")
        else:
            print_warning("No workflow metadata found in image")

    # Show node details if requested
    if args.node:
        print_node_details(prompt, args.node)

    # Show node summary if requested
    if args.nodes or args.nodes_only:
        print_node_summary(prompt)

    # Skip extraction if nodes-only mode
    if args.nodes_only:
        return

    # Require schema for extraction
    if not args.schema:
        print_error("Schema path is required for extraction (or use --nodes-only)")
        parser.print_help()
        sys.exit(1)

    # Resolve schema path
    schema_path = resolve_path(args.schema, SCHEMA_DIR, "Schema")
    print_success(f"Found schema: {schema_path}")

    # Attempt extraction
    print_info("Attempting extraction with schema...")
    try:
        extracted = extract_from_prompt(prompt, schema_path)
        print_success("Extraction successful!")
        print_extraction_result(extracted)

    except Exception as e:
        print_detailed_error(e, prompt, schema_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
