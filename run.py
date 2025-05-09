#!/usr/bin/env python
"""
Run Employee Engagement Dashboard
===============================
This script runs the employee engagement analysis and launches the Streamlit dashboard.

Author: Claude
Date: May 9, 2025
"""

import os
import argparse
import subprocess
import time

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "nltk",
        "wordcloud",
        "networkx",
        "scikit-learn",
        "streamlit",
        "plotly",
        "torch",
        "transformers",
        "scipy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        install = input("Do you want to install these packages now? (y/n): ")
        
        if install.lower() == 'y':
            subprocess.run(["pip", "install"] + missing_packages)
            print("Dependencies installed.")
            return True
        else:
            print("Please install the missing packages before running the dashboard.")
            return False
    
    return True

def run_analysis(force_rerun=False):
    """Run the employee engagement analysis if needed."""
    # Check if analysis has already been run
    if not force_rerun and os.path.exists('output/data/survey_data_clean.csv') and os.path.exists('output/synthesis_results.json'):
        print("Analysis results already exist. Use --force to rerun the analysis.")
        return True
    
    print("Running employee engagement analysis...")
    
    try:
        # Import and run the analysis
        from employee_engagement_analysis import run_employee_engagement_analysis
        
        # Create output directories if they don't exist
        os.makedirs('output/figures', exist_ok=True)
        os.makedirs('output/data', exist_ok=True)
        
        # Run the analysis
        results = run_employee_engagement_analysis()
        print("Analysis completed successfully.")
        return True
    except Exception as e:
        print(f"Error running analysis: {e}")
        return False

def run_streamlit_dashboard():
    """Run the Streamlit dashboard."""
    print("Starting Streamlit dashboard...")
    
    try:
        subprocess.run(["streamlit", "run", "employee_engagement_llm_app.py"])
    except Exception as e:
        print(f"Error running Streamlit dashboard: {e}")

def main():
    """Main function to run the analysis and dashboard."""
    parser = argparse.ArgumentParser(description="Run Employee Engagement Dashboard")
    parser.add_argument("--force", action="store_true", help="Force rerun of analysis even if results exist")
    parser.add_argument("--analysis-only", action="store_true", help="Run only the analysis without launching the dashboard")
    parser.add_argument("--dashboard-only", action="store_true", help="Run only the dashboard without running the analysis")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Run the analysis if needed
    if not args.dashboard_only:
        if not run_analysis(args.force):
            print("Analysis failed. Cannot proceed.")
            return
    
    # Run the dashboard if requested
    if not args.analysis_only:
        run_streamlit_dashboard()

if __name__ == "__main__":
    main()
