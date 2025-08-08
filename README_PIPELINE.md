# AI Research Pipeline - Complete Implementation

This is a comprehensive AI research pipeline that automatically performs data science research from start to finish. The pipeline uses 13 specialized agents to discover datasets, clean data, perform analysis, build models, and generate insights.

## 🚀 Quick Start

### 1. Setup Environment

First, create a `.env` file in the project root with your Kaggle API credentials:

```bash
# .env file
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

You can get your Kaggle API key from: https://www.kaggle.com/settings/account

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

#### Option A: Use the simple runner script
```bash
python run_research_pipeline.py --question "What factors influence customer satisfaction in e-commerce?"
```

#### Option B: Use the test script
```bash
python test_pipeline_complete.py
```

#### Option C: Use the main script directly
```bash
python main.py --question "Your research question here"
```

## 📋 Pipeline Overview

The pipeline executes 13 agents in the exact order specified below:

### 1. **Dataset Discovery Agent** (HGFT: BAAI/bge-base-en-v1.5)
- Finds semantically relevant datasets from Kaggle
- Uses dense embeddings to match research questions to datasets
- Updates `context_chain.dataset_discovery`

### 2. **Data Acquisition Agent**
- Downloads selected datasets using Kaggle API
- Unzips files to `data/raw/`
- Handles large file downloads with streaming
- Updates `context_chain.data_acquisition`

### 3. **Data Quality Agent**
- Removes duplicate rows
- Handles missing values with imputation strategies
- Ensures consistent data types
- Saves cleaned data to `data/cleaned/`
- Updates `context_chain.data_quality`

### 4. **Documentation Agent** (HGFT: facebook/bart-large-cnn)
- Generates readable dataset documentation
- Creates column descriptions and summaries
- Uses BART for concise, human-friendly descriptions
- Updates `context_chain.documentation`

### 5. **EDA Agent** (Exploratory Data Analysis)
- Calculates descriptive statistics
- Generates correlation matrices
- Creates histograms, boxplots, and scatter plots
- Saves outputs to `output/eda_01/`
- Updates `context_chain.eda`

### 6. **Feature Engineering Agent**
- Encodes categorical variables
- Scales numeric variables
- Generates new features (ratios, interactions, binning)
- Saves transformed data to `data/processed/`
- Updates `context_chain.feature_engineering`

### 7. **Statistical Analysis Agent**
- Performs hypothesis tests (t-tests, ANOVA)
- Runs correlation tests (Pearson/Spearman)
- Records p-values and effect sizes
- Updates `context_chain.statistical_analysis`

### 8. **Model Architecture Agent**
- Detects problem type (classification/regression/clustering)
- Recommends suitable ML algorithms
- Provides initial parameter suggestions
- Updates `context_chain.model_architecture`

### 9. **Hyperparameter Optimization Agent**
- Performs grid search, random search, or Bayesian optimization
- Tracks best parameters and validation scores
- Updates `context_chain.hyperparameter_optimization`

### 10. **Model Validation Agent**
- Splits data into train/test sets
- Trains models and calculates metrics
- Saves confusion matrices and error plots
- Updates `context_chain.model_validation`

### 11. **Insight Synthesis Agent** (HGFT: google/flan-t5-large)
- Converts raw results into natural language insights
- Links findings to the research question
- Uses instruction-tuned model for coherent prose
- Updates `context_chain.insight_synthesis`

### 12. **Visualization Agent**
- Compiles plots from EDA, statistical analysis, and model validation
- Adds captions describing each plot
- Saves combined visuals to `output/visualization_01/`
- Updates `context_chain.visualization`

### 13. **Final Report Generator**
- Produces formal PDF research paper
- Includes Abstract, Introduction, Methods, Results, Conclusion
- Exports to `output/reports/`
- Updates `context_chain.final_report`

## 📁 Output Structure

```
output/
├── reports/                    # Final PDF research reports
├── eda_01/                    # Exploratory data analysis results
├── visualization_01/           # Charts and plots
├── data_quality_01/           # Data cleaning results
├── documentation_01/           # Dataset documentation
├── feature_engineering_01/     # Feature transformation results
├── statistical_analysis_01/    # Statistical test results
├── model_architecture_01/      # Model selection results
├── hyperparameter_optimization_01/  # Optimization results
├── model_validation_01/        # Model performance results
├── insight_synthesis_01/       # Generated insights
└── pipeline_results/           # Pipeline execution logs

data/
├── raw/                       # Downloaded datasets
├── cleaned/                    # Cleaned datasets
└── processed/                  # Feature-engineered datasets

context/
└── context_output.json         # Pipeline state and results
```

## 🔧 Configuration

### Environment Variables

The pipeline requires these environment variables in your `.env` file:

```bash
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### Hugging Face Transformers

The pipeline uses three specific transformer models:

1. **BAAI/bge-base-en-v1.5** - For semantic dataset discovery
2. **facebook/bart-large-cnn** - For dataset documentation summarization
3. **google/flan-t5-large** - For insight synthesis

These models are automatically downloaded on first use.

## 📊 Context Schema

The pipeline maintains state in `context/context_output.json`:

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

## 🛠️ Usage Examples

### Basic Usage
```bash
python run_research_pipeline.py --question "What factors predict customer churn?"
```

### Verbose Logging
```bash
python run_research_pipeline.py --question "How do weather patterns affect sales?" --verbose
```

### Custom Research Questions
```bash
python run_research_pipeline.py --question "What are the key drivers of employee satisfaction in tech companies?"
```

## 🔍 Monitoring and Debugging

### Check Pipeline Status
```bash
python main.py --status
```

### View Pipeline Logs
```bash
cat context/context_output.json | jq '.pipeline_log'
```

### Check Agent Results
```bash
cat context/context_output.json | jq '.context_chain'
```

## 🚨 Error Handling

The pipeline is designed to be fault-tolerant:

- Each agent handles exceptions gracefully
- Failed agents don't stop the pipeline
- Errors are logged in the context
- The pipeline continues with the next agent
- Final report includes success/failure summary

## 📈 Performance Considerations

- **Dataset Size**: Large datasets may take longer to process
- **Model Downloads**: First run downloads transformer models (~2-3GB)
- **Memory Usage**: EDA and visualization can be memory-intensive
- **Network**: Dataset downloads depend on internet speed

## 🔧 Troubleshooting

### Common Issues

1. **Kaggle API Error**: Verify your `.env` file has correct credentials
2. **Memory Error**: Reduce dataset size or use smaller models
3. **Download Timeout**: Check internet connection and Kaggle API status
4. **JSON Serialization Error**: Check for numpy types in agent outputs

### Debug Mode
```bash
python run_research_pipeline.py --question "test" --verbose
```

## 📚 Dependencies

Key dependencies include:
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `transformers` - Hugging Face models
- `kaggle` - Dataset API
- `reportlab` - PDF generation
- `matplotlib/seaborn` - Visualization
- `python-dotenv` - Environment variables

## 🤝 Contributing

The pipeline is modular and extensible. Each agent follows the same interface:

```python
class MyAgent:
    def __init__(self):
        # Initialize agent
        pass
    
    def execute(self) -> Dict[str, Any]:
        # Execute agent logic
        # Read context, process data, update context
        pass
```

## 📄 License

This project is open source. See LICENSE for details.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the pipeline logs in `context/context_output.json`
3. Enable verbose logging with `--verbose` flag
4. Check the output directories for detailed results
