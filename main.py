import os
import json
import pandas as pd
import logging
from app.model_framework import ExcelLoader, LLMClientConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_trust_game_experiment(excel_path: str, output_dir: str):
    """
    Main function to run trust game experiments
    
    Args:
        excel_path: Path to the Excel configuration file
        output_dir: Directory to save results
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configurations from Excel
        logger.info(f"Loading configurations from {excel_path}")
        excel_loader = ExcelLoader(excel_path)
        configs = excel_loader.load()
        
        if not configs:
            logger.error("No valid configurations found in Excel file")
            return
        
        # Process each configuration
        for idx, config in enumerate(configs):
            logger.info(f"Processing configuration {idx + 1}/{len(configs)}")
            
            try:
                # Run model and get results
                results = config.run_model(os.getenv("openrouter_token"))
                
                # Save JSON results
                json_filename = os.path.join(output_dir, f"results_config_{idx + 1}.json")
                config.save_json(json_filename, results)
                
                # Save Excel results
                excel_filename = os.path.join(output_dir, f"results_config_{idx + 1}.xlsx")
                config.save_excel(excel_filename, results)
                
            except Exception as e:
                logger.error(f"Error processing configuration {idx + 1}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Critical error in experiment: {e}")

def load_environment_vars():
    """
    Load environment variables and validate they exist
    """
    try:
        # These will be loaded from .env by model_framework
        openrouter_token = os.getenv("openrouter_token")
        
        if not openrouter_token:
            logger.warning("OpenRouter token not found in environment variables")
            
        return {
            "openrouter_token": openrouter_token
        }
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return {}

# Example usage
if __name__ == '__main__':
    # Load environment variables
    load_environment_vars()
    
    # Run the trust game experiment
    excel_file = "config.xlsx"
    output_directory = "results"
    
    if os.path.exists(excel_file):
        logger.info(f"Starting trust game experiment with {excel_file}")
        run_trust_game_experiment(excel_file, output_directory)
    else:
        logger.error(f"Excel file {excel_file} does not exist")
        print(f"Error: Excel file {excel_file} not found. Please create a configuration file.")
        print("Using sample Python script functionality:")

