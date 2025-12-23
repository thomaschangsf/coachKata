#!/usr/bin/env python3
"""
Main entry point for pickleball pose analysis.

This script orchestrates model loading, image processing, and scoring
for pickleball pose analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .image_processor import process_three_positions
from .model_loader import load_sam3d_model
from .scoring import (
    score_contact_position,
    score_finish_position,
    score_preparation_position,
)


def analyze_three_positions(
    preparation_path: str,
    contact_path: str,
    finish_path: str,
    output_path: str | None = None,
    hf_repo_id: str = "facebook/sam-3d-body-dinov3",
    device: str = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Analyze three pickleball positions and return scores.

    Args:
        preparation_path: Path to preparation position image
        contact_path: Path to contact position image
        finish_path: Path to finish position image
        output_path: Optional path to save JSON results
        hf_repo_id: HuggingFace repository ID for model
        device: Device to use ("auto", "cuda", or "cpu")
        verbose: Whether to print progress

    Returns:
        Dictionary containing pose data and scores for all three positions
    """
    if verbose:
        print("=" * 60)
        print("Pickleball Pose Analysis - Phase 1")
        print("=" * 60)
        print()

    # Load model
    if verbose:
        print("Loading SAM 3D Body model...")
    estimator, config = load_sam3d_model(
        hf_repo_id=hf_repo_id,
        device=device,
    )

    # Process images
    if verbose:
        print("\nProcessing images...")
    results = process_three_positions(
        estimator,
        preparation_path,
        contact_path,
        finish_path,
    )

    # Score each position
    if verbose:
        print("\nScoring positions...")

    prep_score = score_preparation_position(results['preparation'])
    contact_score = score_contact_position(results['contact'])
    finish_score = score_finish_position(results['finish'])

    # Combine results
    analysis_results = {
        'preparation': {
            **results['preparation'],
            **prep_score,
        },
        'contact': {
            **results['contact'],
            **contact_score,
        },
        'finish': {
            **results['finish'],
            **finish_score,
        },
        'summary': {
            'preparation_score': prep_score['preparation_score'],
            'contact_score': contact_score['contact_score'],
            'finish_score': finish_score['finish_score'],
            'cumulative_score': (
                prep_score['preparation_score'] +
                contact_score['contact_score'] +
                finish_score['finish_score']
            ) / 3.0,
        },
        'config': config,
    }

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Preparation Score: {prep_score['preparation_score']:.1f}/100")
        print(f"Contact Score:     {contact_score['contact_score']:.1f}/100")
        print(f"Finish Score:       {finish_score['finish_score']:.1f}/100")
        print(f"Cumulative Score:   {analysis_results['summary']['cumulative_score']:.1f}/100")
        print("=" * 60)

    # Save to file if requested
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy arrays to lists."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj

        serializable_results = convert_to_serializable(analysis_results)

        with open(output_path_obj, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        if verbose:
            print(f"\nResults saved to: {output_path_obj}")

    return analysis_results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze pickleball poses from three images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python analyze_positions.py prep.jpg contact.jpg finish.jpg

  # Save results to JSON
  python analyze_positions.py prep.jpg contact.jpg finish.jpg -o results.json

  # Use specific model
  python analyze_positions.py prep.jpg contact.jpg finish.jpg --model facebook/sam-3d-body-vith

  # Force CPU usage
  python analyze_positions.py prep.jpg contact.jpg finish.jpg --device cpu
        """
    )

    parser.add_argument(
        'preparation',
        type=str,
        help='Path to preparation position image'
    )
    parser.add_argument(
        'contact',
        type=str,
        help='Path to contact position image'
    )
    parser.add_argument(
        'finish',
        type=str,
        help='Path to finish position image'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output JSON file path (optional)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='facebook/sam-3d-body-dinov3',
        help='HuggingFace model repository ID (default: facebook/sam-3d-body-dinov3)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate input files
    for path_name, path_value in [
        ('preparation', args.preparation),
        ('contact', args.contact),
        ('finish', args.finish),
    ]:
        if not Path(path_value).exists():
            print(f"Error: {path_name} image not found: {path_value}", file=sys.stderr)
            sys.exit(1)

    try:
        analyze_three_positions(
            preparation_path=args.preparation,
            contact_path=args.contact,
            finish_path=args.finish,
            output_path=args.output,
            hf_repo_id=args.model,
            device=args.device,
            verbose=not args.quiet,
        )

        # Exit with code 0 on success
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
