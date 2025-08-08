# AI Research Pipeline

A comprehensive AI-powered research pipeline that automatically discovers, analyzes, and documents datasets using Hugging Face Transformers and semantic search.

## Overview

This pipeline implements the master specification for an AI research system that:

1. **Discovers datasets** using semantic search with BAAI/bge-base-en-v1.5 embeddings
2. **Downloads and prepares** data using Kaggle API
3. **Cleans and validates** data quality
4. **Generates documentation** using facebook/bart-large-cnn summarization
5. **Executes in exact order** without skipping steps
6. **Maintains cumulative context** through standardized JSON files

## Pipeline Architecture

The pipeline executes agents in the following order:

1. **Dataset Discovery Agent** - Finds relevant datasets using semantic search
2. **Data Acquisition Agent** - Downloads datasets from Kaggle
3. **Data Quality Agent** - Cleans and validates data
4. **Documentation Agent** - Generates dataset documentation
5. **EDA Agent** - Exploratory data analysis (planned)
6. **Feature Engineering Agent** - Feature transformation (planned)
7. **Statistical Analysis Agent** - Hypothesis testing (planned)
8. **Model Architecture Agent** - Model selection (planned)
9. **Hyperparameter Optimization Agent** - Parameter tuning (planned)
10. **Model Validation Agent** - Model evaluation (planned)
11. **Insight Synthesis Agent** - Natural language insights (planned)
12. **Visualization Agent** - Plot generation (planned)

## Features

### Implemented

- **Standardized Context Management**: All agents read/write to `context/context_output.json`
- **Semantic Dataset Discovery**: Uses BAAI/bge-base-en-v1.5 for semantic search
- **Kaggle Integration**: Automatic dataset downloading with authentication
- **Data Quality Pipeline**: Duplicate removal, missing value handling, type correction
- **AI-Powered Documentation**: Uses facebook/bart-large-cnn for dataset summarization
- **Error Handling**: Graceful failure handling with context logging
- **Pipeline Orchestration**: Sequential execution with status tracking

### In Progress

- EDA with automated plotting - V1.0, working but needs change
- Feature engineering and transformation - working
- Statistical analysis and hypothesis testing - errors in calculation, need to refer back to data pre-processing
- Model architecture selection - should work after previous
- Hyperparameter optimization - does not work, pipeline stops here
- Model validation and metrics
- Insight synthesis with google/flan-t5-large
- Visualization compilation
- Final PDF report generation - working

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AgenticAIProject
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle credentials**:
   Create a `.env` file in the project root:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

4. **Get Kaggle API credentials**:
   - Go to [Kaggle Settings](https://www.kaggle.com/account)
   - Scroll to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json` and extract credentials

## Usage

### Quick Start

Run the pipeline with a research question:

```bash
python main.py --research-question "Analyze customer satisfaction data"
```

### Check Pipeline Status

```bash
python main.py --status
```

### Test the Pipeline

```bash
python test_pipeline.py
```

## Project Structure

```
AgenticAIProject/
├── context/
│   └── context_output.json          # Standardized context file
├── data/
│   ├── raw/                         # Downloaded datasets
│   └── cleaned/                     # Processed datasets
├── output/
│   ├── dataset_discovery_01/        # Discovery results
│   ├── data_acquisition_01/         # Download results
│   ├── data_quality_01/             # Quality results
│   ├── documentation_01/            # Documentation results
│   └── pipeline_results/            # Pipeline execution results
├── src/
│   ├── agents/                      # All pipeline agents
│   └── core/
│       └── context_manager.py       # Context management utilities
├── logs/
│   └── pipeline.log                 # Pipeline execution logs
├── main.py                          # Pipeline orchestrator
├── test_pipeline.py                 # Test script
└── requirements.txt                 # Dependencies
```

## Context Schema

All agents use a standardized context structure:

```json
{
  "project_metadata": {
    "research_question": "string",
    "created_at": "ISO timestamp",
    "dataset_refs": ["string"]
  },
  "context_chain": {
    "dataset_discovery": {},
    "data_acquisition": {},
    "data_quality": {},
    "documentation": {},
    "eda": {},
    "feature_engineering": {},
    "statistical_analysis": {},
    "model_architecture": {},
    "hyperparameter_optimization": {},
    "model_validation": {},
    "insight_synthesis": {},
    "visualization": {},
    "final_report": {}
  },
  "pipeline_log": [
    {
      "agent": "string",
      "timestamp": "ISO timestamp",
      "message": "string"
    }
  ]
}
```

## Agent Details

### Dataset Discovery Agent
- **Model**: BAAI/bge-base-en-v1.5
- **Purpose**: Semantic search for relevant datasets
- **Output**: Selected datasets with similarity scores

### Data Acquisition Agent
- **API**: Kaggle API
- **Purpose**: Download and prepare datasets
- **Output**: Local file paths and metadata

### Data Quality Agent
- **Purpose**: Clean and validate data
- **Features**: Duplicate removal, missing value handling, type correction
- **Output**: Cleaned datasets and quality metrics

### Documentation Agent
- **Model**: facebook/bart-large-cnn
- **Purpose**: Generate dataset documentation
- **Features**: Column analysis, sample data, AI-powered summaries
- **Output**: Comprehensive dataset documentation

## Configuration

### Environment Variables

Create a `.env` file with:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### Output Directories

The pipeline automatically creates:

- `context/` - Context files
- `data/raw/` - Downloaded datasets
- `data/cleaned/` - Processed datasets
- `output/` - Agent outputs
- `logs/` - Execution logs

## Error Handling

- Each agent handles exceptions gracefully
- Failed agents log errors to context
- Pipeline continues with remaining agents
- Detailed error logs in `logs/pipeline.log`

## Logging

The pipeline provides comprehensive logging:

- **File logging**: `logs/pipeline.log`
- **Context logging**: All steps logged to `context/context_output.json`
- **Agent-specific logs**: Each agent logs to its output directory

## Testing

Run the test suite:

```bash
python test_pipeline.py
```

This will:
1. Check environment setup
2. Test individual agents
3. Test pipeline orchestration
4. Verify context management

## Contributing

1. Follow the master specification for agent implementation
2. Use standardized context management
3. Implement proper error handling
4. Add comprehensive logging
5. Include tests for new agents

## License

[Add your license information here]

## Support

For issues and questions:
1. Check the logs in `logs/pipeline.log`
2. Verify Kaggle credentials in `.env`
3. Run `python test_pipeline.py` to diagnose issues

4. Check the context file at `context/context_output.json` 
