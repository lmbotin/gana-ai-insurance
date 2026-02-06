#!/usr/bin/env python3
"""
CLI for parsing Track & Trace / AI Operational Liability claims.

Usage:
    python -m src.fnol.cli --text "incident description" --images log1.json log2.json
    python -m src.fnol.cli --text-file fixtures/incident01.txt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from .config import ExtractionConfig
from .pipeline import parse_claim


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def read_text_file(path: str) -> str:
    """Read text from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Parse Track & Trace / AI Operational Liability claims',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse with inline text
  python -m src.fnol.cli --text "Shipment delayed 48 hours due to routing model failure" --pretty

  # Parse with text file
  python -m src.fnol.cli --text-file fixtures/incident01.txt

  # Parse with claimant info
  python -m src.fnol.cli --text "Package lost at HUB-LAX-03" --claimant-name "Acme Corp" --policy-number POL-TT-123

  # Use mock LLM (no API key needed)
  python -m src.fnol.cli --text "AI prediction failure caused misroute" --llm-provider mock

  # Pretty print output
  python -m src.fnol.cli --text "System outage at sorting facility" --pretty
        """
    )

    # Text input (mutually exclusive)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        '--text',
        type=str,
        help='Claim description text (inline)'
    )
    text_group.add_argument(
        '--text-file',
        type=str,
        help='Path to file containing claim description'
    )

    # Image inputs
    parser.add_argument(
        '--images',
        nargs='*',
        default=[],
        help='Paths to images (space-separated)'
    )

    # Claimant info
    parser.add_argument(
        '--claimant-name',
        type=str,
        help='Claimant name'
    )
    parser.add_argument(
        '--policy-number',
        type=str,
        help='Policy number'
    )
    parser.add_argument(
        '--contact-phone',
        type=str,
        help='Contact phone'
    )
    parser.add_argument(
        '--contact-email',
        type=str,
        help='Contact email'
    )

    # Configuration
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['claude', 'openai', 'mock'],
        default='mock',
        help='LLM provider to use (default: mock for testing without API key)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        help='Specific LLM model to use'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for LLM provider (or set ANTHROPIC_API_KEY/OPENAI_API_KEY env var)'
    )

    # Output options
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file path (default: print to stdout)'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output'
    )

    # Logging
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Get text
        if args.text:
            text = args.text
        else:
            text = read_text_file(args.text_file)

        logger.info(f"Input text: {len(text)} characters")

        # Get images
        image_paths: List[str] = args.images or []
        logger.info(f"Input images: {len(image_paths)} files")

        # Build claimant info
        claimant_info = {}
        if args.claimant_name:
            claimant_info['name'] = args.claimant_name
        if args.policy_number:
            claimant_info['policy_number'] = args.policy_number
        if args.contact_phone:
            claimant_info['contact_phone'] = args.contact_phone
        if args.contact_email:
            claimant_info['contact_email'] = args.contact_email

        # Create config
        config = ExtractionConfig(
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            api_key=args.api_key
        )

        logger.info(f"Using LLM provider: {config.llm_provider}, model: {config.llm_model}")

        # Validate config
        if not config.validate():
            logger.error(
                f"Invalid configuration: {config.llm_provider} requires API key. "
                "Set --api-key or environment variable."
            )
            sys.exit(1)

        # Parse claim
        logger.info("Starting claim parsing...")
        claim = parse_claim(
            text=text,
            image_paths=image_paths,
            claimant_info=claimant_info if claimant_info else None,
            config=config
        )

        logger.info(f"Claim parsed successfully: {claim.claim_id}")

        # Convert to JSON (Pydantic v2 compatible)
        indent = 2 if args.pretty else None
        json_output = claim.model_dump_json(indent=indent)

        # Output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            logger.info(f"Output written to: {output_path}")
        else:
            print(json_output)

        # Print summary
        if args.verbose:
            print("\n" + "="*60, file=sys.stderr)
            print("EXTRACTION SUMMARY", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(f"Claim ID: {claim.claim_id}", file=sys.stderr)
            print(f"Incident Type: {claim.incident.incident_type.value}", file=sys.stderr)
            print(f"Asset Type: {claim.operational_impact.asset_type.value}", file=sys.stderr)
            print(f"Impact Severity: {claim.operational_impact.impact_severity.value}", file=sys.stderr)
            print(f"System Logs: {claim.evidence.system_log_count}", file=sys.stderr)
            print(f"Missing Evidence: {len(claim.evidence.missing_evidence)}", file=sys.stderr)
            print(f"Conflicts: {len(claim.consistency.conflict_details)}", file=sys.stderr)
            print("="*60, file=sys.stderr)

    except Exception as e:
        logger.error(f"Error parsing claim: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
