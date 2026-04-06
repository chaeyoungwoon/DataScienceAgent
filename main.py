"""
AI Research Pipeline - Main Orchestrator

Runs all agents in sequence. Each agent reads context/context_output.json,
modifies only its relevant keys, and writes back the updated file.

Usage:
    python main.py --research-question "What factors predict house prices?"
    python main.py --research-question "..." --start-from eda
    python main.py --status
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.context_manager import (
    read_context, write_context, log_step, set_research_question
)
from src.agents.dataset_discovery_agent import DatasetDiscoveryAgent
from src.agents.data_acquisition_agent import DataAcquisitionAgent
from src.agents.data_quality_agent import DataQualityAgent
from src.agents.documentation_agent import DocumentationAgent
from src.agents.eda_agent_simple import EDAAgent
from src.agents.feature_engineering_agent_simple import FeatureEngineeringAgent
from src.agents.statistical_analysis_agent_simple import StatisticalAnalysisAgent
from src.agents.model_architecture_agent_simple import ModelArchitectureAgent
from src.agents.hyperparameter_optimization_agent_simple import HyperparameterOptimizationAgent
from src.agents.model_validation_agent_simple import ModelValidationAgent
from src.agents.insight_synthesis_agent_simple import InsightSynthesisAgent
from src.agents.visualization_agent_simple import VisualizationAgent
from src.agents.final_report_generator import FinalReportGenerator


PIPELINE_STAGES = [
    ('dataset_discovery',           DatasetDiscoveryAgent),
    ('data_acquisition',            DataAcquisitionAgent),
    ('data_quality',                DataQualityAgent),
    ('documentation',               DocumentationAgent),
    ('eda',                         EDAAgent),
    ('feature_engineering',         FeatureEngineeringAgent),
    ('statistical_analysis',        StatisticalAnalysisAgent),
    ('model_architecture',          ModelArchitectureAgent),
    ('hyperparameter_optimization', HyperparameterOptimizationAgent),
    ('model_validation',            ModelValidationAgent),
    ('insight_synthesis',           InsightSynthesisAgent),
    ('visualization',               VisualizationAgent),
    ('final_report',                FinalReportGenerator),
]

STAGE_NAMES = [name for name, _ in PIPELINE_STAGES]


class PipelineOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results: Dict[str, Any] = {
            'start_time': None,
            'end_time': None,
            'total_agents': len(PIPELINE_STAGES),
            'successful_agents': 0,
            'failed_agents': 0,
            'skipped_agents': 0,
            'agent_results': {},
            'pipeline_status': 'not_started',
        }

    def run_pipeline(self, research_question: str, start_from: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete pipeline, optionally resuming from a specific agent."""
        self._ensure_dirs()

        context = read_context()
        set_research_question(context, research_question)
        log_step(context, 'orchestrator', f"Pipeline started: {research_question}")
        write_context(context)

        self.results['start_time'] = datetime.now().isoformat()
        self.results['pipeline_status'] = 'running'

        start_idx = 0
        if start_from:
            if start_from not in STAGE_NAMES:
                raise ValueError(f"Unknown stage '{start_from}'. Valid stages: {STAGE_NAMES}")
            start_idx = STAGE_NAMES.index(start_from)
            self.logger.info(f"Resuming pipeline from: {start_from}")

        total = len(PIPELINE_STAGES)
        for i, (agent_name, agent_class) in enumerate(PIPELINE_STAGES):
            if i < start_idx:
                self._print_stage(i + 1, total, agent_name, 'SKIP')
                self.results['agent_results'][agent_name] = {'status': 'skipped'}
                self.results['skipped_agents'] += 1
                continue

            self._print_stage(i + 1, total, agent_name, 'RUN')
            try:
                agent = agent_class()
                result = agent.execute()
                self.results['agent_results'][agent_name] = {
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                }
                self.results['successful_agents'] += 1
                self._print_stage(i + 1, total, agent_name, 'OK')
            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {e}", exc_info=True)
                self.results['agent_results'][agent_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                }
                self.results['failed_agents'] += 1
                self._print_stage(i + 1, total, agent_name, 'FAIL', str(e))

                ctx = read_context()
                log_step(ctx, 'orchestrator', f"Agent {agent_name} failed: {e}")
                write_context(ctx)

        self.results['end_time'] = datetime.now().isoformat()
        self.results['pipeline_status'] = (
            'completed' if self.results['failed_agents'] == 0 else 'completed_with_errors'
        )

        ctx = read_context()
        log_step(ctx, 'orchestrator',
                 f"Pipeline finished. OK={self.results['successful_agents']} "
                 f"FAIL={self.results['failed_agents']} SKIP={self.results['skipped_agents']}")
        write_context(ctx)

        self._save_results()
        return self.results

    def _print_stage(self, idx: int, total: int, name: str, status: str, detail: str = ''):
        """Print a single-line status update for a pipeline stage."""
        status_symbols = {
            'RUN':  '\033[33m●\033[0m',   # yellow
            'OK':   '\033[32m✓\033[0m',   # green
            'FAIL': '\033[31m✗\033[0m',   # red
            'SKIP': '\033[90m–\033[0m',   # grey
        }
        sym = status_symbols.get(status, ' ')
        prefix = f"[{idx:02d}/{total}]"
        line = f"  {sym} {prefix} {name}"
        if detail and status == 'FAIL':
            line += f"\n       Error: {detail[:120]}"
        if status == 'RUN':
            print(line, end='\r', flush=True)
        else:
            print(line + '          ')  # overwrite spinner line

    def _save_results(self):
        results_dir = Path("output/pipeline_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "pipeline_execution_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        ctx = read_context()
        with open(results_dir / "final_context.json", 'w') as f:
            json.dump(ctx, f, indent=2, default=str)

    def _ensure_dirs(self):
        for d in ['logs', 'context', 'data/raw', 'data/cleaned', 'data/processed', 'output']:
            Path(d).mkdir(parents=True, exist_ok=True)


def print_status():
    """Print a human-readable pipeline status from the context file."""
    ctx = read_context()
    meta = ctx.get('project_metadata', {})
    log = ctx.get('pipeline_log', [])
    chain = ctx.get('context_chain', {})

    print("\n=== Pipeline Status ===")
    print(f"Research Question : {meta.get('research_question', '(not set)')}")
    print(f"Started           : {meta.get('created_at', 'unknown')}\n")

    print("Agent Results:")
    for stage_name in STAGE_NAMES:
        data = chain.get(stage_name, {})
        if not data:
            state = '\033[90m pending\033[0m'
        elif data.get('status') == 'failed' or 'error' in data:
            state = f'\033[31m FAILED\033[0m  – {data.get("error", "")[:80]}'
        else:
            state = '\033[32m done\033[0m'
        print(f"  {stage_name:<35} {state}")

    if log:
        print("\nRecent Log (last 5 entries):")
        for entry in log[-5:]:
            ts = entry.get('timestamp', '')[:19]
            print(f"  [{ts}] {entry.get('agent', '?')}: {entry.get('message', '')}")
    print()


def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler('logs/pipeline.log'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    # Suppress noisy third-party loggers
    for noisy in ['transformers', 'sentence_transformers', 'torch', 'PIL']:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description='AI Research Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -r "What predicts house prices?"
  python main.py -r "Titanic survival factors" --start-from eda
  python main.py --status
        """,
    )
    parser.add_argument('--research-question', '-r',
                        help='The research question to investigate')
    parser.add_argument('--start-from', '-f', metavar='STAGE',
                        choices=STAGE_NAMES,
                        help=f'Resume pipeline from this stage (choices: {", ".join(STAGE_NAMES)})')
    parser.add_argument('--status', '-s', action='store_true',
                        help='Show current pipeline status and exit')
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if not args.research_question:
        parser.error('--research-question is required unless --status is used')

    setup_logging()

    print(f"\n{'='*60}")
    print(f"  AI Research Pipeline")
    print(f"  Question: {args.research_question}")
    if args.start_from:
        print(f"  Resuming from: {args.start_from}")
    print(f"{'='*60}\n")

    orchestrator = PipelineOrchestrator()
    try:
        results = orchestrator.run_pipeline(args.research_question, start_from=args.start_from)
    except Exception as e:
        print(f"\n\033[31mPipeline failed: {e}\033[0m")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Pipeline {results['pipeline_status'].upper()}")
    print(f"  ✓ Successful : {results['successful_agents']}")
    if results['failed_agents']:
        print(f"  ✗ Failed     : {results['failed_agents']}")
    if results['skipped_agents']:
        print(f"  – Skipped    : {results['skipped_agents']}")
    print(f"  Results      : output/pipeline_results/")
    print(f"  Report       : output/reports/")
    print(f"{'='*60}\n")

    if results['failed_agents']:
        print("Failed agents:")
        for name, r in results['agent_results'].items():
            if r.get('status') == 'failed':
                print(f"  • {name}: {r.get('error', 'unknown error')[:120]}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
