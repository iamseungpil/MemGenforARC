#!/usr/bin/env python3
"""
Config-based Pipeline Runner for MemGen Experiments
====================================================

This script reads experiment configurations from YAML files and executes
the pipeline stages in sequence with automatic checkpoint discovery.

Usage:
    python run_pipeline.py --config configs/full_pipeline.yaml
    python run_pipeline.py --config configs/weaver_grpo_only.yaml --dry-run
    python run_pipeline.py --config configs/ltpo_sweep.yaml --stage ltpo_sweep
"""

import argparse
import os
import subprocess
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_name_safe(model_name: str) -> str:
    """Extract model name for filesystem path (last part after /)."""
    return model_name.split('/')[-1]


def find_latest_checkpoint(
    output_root: str,
    dataset: str,
    model_name: str,
    checkpoint_type: str = 'weaver'
) -> Optional[str]:
    """Find the latest checkpoint for a given type."""
    model_name_safe = get_model_name_safe(model_name)
    base_dir = Path(output_root) / 'train' / dataset / model_name_safe

    if not base_dir.exists():
        return None

    # Find latest timestamp directory
    timestamp_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('pn=')],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not timestamp_dirs:
        return None

    latest_dir = timestamp_dirs[0]

    # Find the checkpoint
    if checkpoint_type == 'weaver':
        checkpoint_path = latest_dir / 'weaver' / 'weaver_lora'
    elif checkpoint_type == 'trigger':
        checkpoint_path = latest_dir / 'trigger' / 'trigger_lora'
    else:
        return None

    if checkpoint_path.exists():
        return str(checkpoint_path)

    return None


def run_stage(
    stage_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    script_dir: Path,
    dry_run: bool = False
) -> bool:
    """Run a single pipeline stage."""
    stage_name = stage_config['name']
    script_name = stage_config.get('script')

    if not stage_config.get('enabled', True):
        logger.info(f"Skipping disabled stage: {stage_name}")
        return True

    logger.info(f"=" * 60)
    logger.info(f"Running stage: {stage_name}")
    logger.info(f"=" * 60)

    # Find the script in model-specific directory
    model_name = experiment_config['model']['name']
    model_name_safe = get_model_name_safe(model_name)
    model_script_dir = script_dir / model_name_safe

    if script_name:
        script_path = model_script_dir / script_name
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        # Build command
        cmd = ['bash', str(script_path)]

        # Add checkpoint path arguments if needed
        if stage_config.get('depends_on'):
            output_root = os.path.expanduser(
                experiment_config.get('output', {}).get('base_dir', '~/data/memgen')
            )
            dataset = experiment_config['model']['dataset']

            weaver_path = find_latest_checkpoint(
                output_root, dataset, model_name, 'weaver'
            )
            if weaver_path:
                cmd.append(weaver_path)

                # For trigger stages, also add trigger path
                if 'trigger' in stage_name and stage_name != 'trigger_grpo':
                    trigger_path = find_latest_checkpoint(
                        output_root, dataset, model_name, 'trigger'
                    )
                    if trigger_path:
                        cmd.append(trigger_path)

        logger.info(f"Command: {' '.join(cmd)}")

        if dry_run:
            logger.info("[DRY RUN] Would execute the above command")
            return True

        # Execute the command
        try:
            result = subprocess.run(
                cmd,
                cwd=str(script_dir.parent.parent),  # Project root
                check=True,
                text=True
            )
            logger.info(f"Stage {stage_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Stage {stage_name} failed with return code {e.returncode}")
            return False
    else:
        logger.warning(f"No script defined for stage: {stage_name}")
        return True


def run_pipeline(
    config: Dict[str, Any],
    script_dir: Path,
    stage_filter: Optional[str] = None,
    dry_run: bool = False
) -> bool:
    """Run the complete pipeline."""
    experiment_name = config.get('experiment', {}).get('name', 'unnamed')
    logger.info(f"Starting pipeline: {experiment_name}")
    logger.info(f"Description: {config.get('experiment', {}).get('description', 'N/A')}")

    stages = config.get('stages', [])

    if stage_filter:
        stages = [s for s in stages if s['name'] == stage_filter]
        if not stages:
            logger.error(f"Stage not found: {stage_filter}")
            return False

    success_count = 0
    total_count = len(stages)

    for stage in stages:
        if run_stage(stage, config, script_dir, dry_run):
            success_count += 1
        else:
            logger.error(f"Pipeline stopped due to failure in stage: {stage['name']}")
            break

    logger.info(f"=" * 60)
    logger.info(f"Pipeline completed: {success_count}/{total_count} stages successful")
    logger.info(f"=" * 60)

    return success_count == total_count


def main():
    parser = argparse.ArgumentParser(
        description='Run MemGen experiment pipeline from YAML config'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration YAML file'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default=None,
        help='Run only a specific stage (by name)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='List all stages in the config and exit'
    )

    args = parser.parse_args()

    # Determine script directory
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / args.config if not os.path.isabs(args.config) else Path(args.config)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Load configuration
    config = load_config(str(config_path))

    # List stages and exit if requested
    if args.list_stages:
        print("\nAvailable stages:")
        print("-" * 40)
        for stage in config.get('stages', []):
            status = "enabled" if stage.get('enabled', True) else "disabled"
            print(f"  - {stage['name']} ({status})")
        print()
        sys.exit(0)

    # Run the pipeline
    success = run_pipeline(
        config,
        script_dir,
        stage_filter=args.stage,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
