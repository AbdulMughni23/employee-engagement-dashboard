# Employee Engagement Analysis & LLM Dashboard

This project provides a comprehensive employee engagement analysis solution that combines quantitative survey data with qualitative focus group insights, and uses a local LLM (Language Model) to generate personalized insights and recommendations.

## Features

- **Comprehensive Data Analysis**: Combines quantitative and qualitative methods to identify key engagement drivers
- **LLM-Powered Insights**: Uses a local Hugging Face language model to generate dynamic insights and recommendations
- **Interactive Dashboard**: Streamlit-based visualization of engagement metrics and insights
- **Department & Role Analysis**: Detailed breakdowns by department and role with specific recommendations
- **Custom Queries**: Ask your own questions about the engagement data and get AI-generated responses

## System Requirements

- Python 3.8+
- 4+ GB RAM (8+ GB recommended for optimal LLM performance)
- 2+ GB disk space (for model download and data storage)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/employee-engagement-analysis.git
   cd employee-engagement-analysis
   ```

2. Run the setup script, which will check and install dependencies:
   ```
   python run_engagement_app.py
   ```

   The script will automatically check for and install the required Python packages, including:
   - pandas, numpy, matplotlib, seaborn
   - nltk, wordcloud, networkx
   - sklearn, scipy
   - streamlit, plotly
   - torch, transformers

## Usage

### Running the Complete System

The easiest way to run the system is with the included runner script:

```
python run_engagement_app.py
```

This will:
1. Check for dependencies and install any missing packages
2. Run the engagement analysis (generating synthetic data if needed)
3. Launch the Streamlit dashboard with LLM capabilities

### Command Line Options

The runner script supports several command-line options:

- `--force`: Force rerun of analysis even if results already exist
- `--analysis-only`: Run only the analysis without launching the dashboard
- `--dashboard-only`: Run only the dashboard without running the analysis

Example:
```
python run_engagement_app.py --dashboard-only
```

### Manual Execution

If you prefer to run the components individually:

1. First, run the analysis:
   ```
   python employee_engagement_analysis.py
   ```

2. Then launch the dashboard:
   ```
   streamlit run employee_engagement_llm_app.py
   ```

## Project Structure

- `employee_engagement_analysis.py`: Main analysis script that processes survey and focus group data
- `employee_engagement_llm_app.py`: Streamlit dashboard with LLM integration
- `focus_group_analysis.py`: Script for qualitative analysis of focus group transcripts
- `run_engagement_app.py`: Helper script to run the complete system
- `output/`: Directory containing analysis results and visualizations
  - `data/`: Processed CSV data files
  - `figures/`: Generated visualizations
  - `qualitative/`: Qualitative analysis results
  - `executive_summary.md`: Summary of key findings
  - `recommendations.md`: Detailed recommendations
  - `synthesis_results.json`: Combined quantitative and qualitative results

## Using the Dashboard

The dashboard includes several pages:

1. **Overview**: Key metrics and insights about overall engagement
2. **Department Analysis**: Detailed metrics for each department with comparisons
3. **Role Comparison**: Compare engagement factors across different roles
4. **LLM Insights**: AI-generated insights on specific topics of interest
5. **Custom Query**: Ask your own questions about the engagement data

## LLM Integration

This system uses TinyLlama, a small language model that can run locally on your machine. The first time you run the dashboard, it will download the model (approximately 2GB). Subsequent runs will use the cached model.

The LLM provides:
- Insights based on analysis data
- Department-specific recommendations
- Comparative analyses between departments or roles
- Responses to custom queries about the engagement data

## Customization

### Using Real Data

By default, the system generates synthetic data for demonstration. To use real data:

1. Replace the synthetic data generation with your own data loading code
2. Place your survey data CSV in `output/data/survey_data_raw.csv`
3. Place focus group transcripts in `output/data/focus_group_*.txt` files

### Customizing the Dashboard

The Streamlit dashboard can be easily modified:
- Edit `employee_engagement_llm_app.py` to add new visualizations or pages
- Modify the CSS styles in the `st.markdown("""<style>...""")` section
- Add new LLM query templates in the "LLM Insights" page section

### Using a Different LLM Model

To use a different Hugging Face model:
1. Change the `model_name` in the `load_llm_model()` function in `employee_engagement_llm_app.py`
2. Adjust the generation parameters (temperature, max_new_tokens, etc.) as needed

## Troubleshooting

### Common Issues

- **"CUDA out of memory"**: The LLM is trying to use GPU but doesn't have enough VRAM. Add `device_map="cpu"` to force CPU usage.
- **Slow LLM responses**: The default model (TinyLlama) is optimized for local use. For faster responses, consider using a smaller model or upgrading hardware.
- **Missing NLTK data**: If you see NLTK errors, manually download required data: `python -m nltk.downloader punkt stopwords`

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Make sure all dependencies are properly installed
3. Try running the analysis and dashboard separately to isolate the issue
4. File an issue on the GitHub repository with details about the problem

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This system uses the TinyLlama model from Hugging Face
- Visualization components built with Plotly and Streamlit
- Analysis leverages pandas, scikit-learn, and NLTK libraries
