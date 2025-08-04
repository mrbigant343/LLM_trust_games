import os
import json
import pandas as pd
import logging
from datetime import datetime
from app.model_framework import ExcelLoader, LLMClientConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_trust_game_experiment(excel_path: str, output_dir: str):
    """
    Main function to run trust game experiments
    
    Args:
        excel_path: Path to the Excel configuration file
        output_dir: D irectory to save results
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize paths and data structures
        checkpoint_path = os.path.join(output_dir, "checkpoints.json")
        merged_json_path = os.path.join(output_dir, "all_results.json")
        merged_json = []
        
        # Load or initialize checkpoints
        checkpoints = {}
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoints = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load checkpoints: {e}")
        
        # Load existing merged JSON data
        if os.path.exists(merged_json_path):
            try:
                with open(merged_json_path, 'r', encoding='utf-8') as f:
                    merged_json = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in existing merged file. Starting fresh.")
        else:
            with open(merged_json_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        # Load configurations from Excel
        logger.info(f"Loading configurations from {excel_path}")
        excel_loader = ExcelLoader(excel_path)
        configs = excel_loader.load()
        
        if not configs:
            logger.error("No valid configurations found in Excel file")
            return
        
        # Generate checkpoints based on model.txt and id column
        for config in configs:
            if not config.id or not config.model:
                continue
            for i in range(1, config.n_requests + 1):
                key = f"{config.id}_{config.model.replace('/', '_')}_{i}"
                if key not in checkpoints:
                    checkpoints[key] = 0
        
        # Save updated checkpoints file
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoints, f, indent=2)
        logger.info(f"Successfully initialized checkpoints at {checkpoint_path}")

        
        # Process each configuration iteratively
        for idx, config in enumerate(configs):
            logger.info(f"Processing configuration {idx + 1}/{len(configs)}")
            config_results = []
            processed = False
            
            try:
                for i in range(config.n_requests):
                    key = f"{config.id}_{config.model.replace('/', '_')}_{i + 1}"
                    
                    # Skip if already completed
                    if checkpoints.get(key) == 1:
                        logger.info(f"Skipping completed {key}")
                        continue
                    
                    processed = True
                    logger.info(f"Processing iteration {i+1} for {config.model}")
                    
                    try:
                        # Generate prompt for this iteration
                        #logger.info(f"Processing request {i+1}/{config.n_requests} for {config.model}")
                        def _call_model():
                            return config.check_and_run_model()

                        # Call model and parse response with retry
                        response_text = config._retry_on_failure(_call_model)
                        result = config.parse_json(response_text)
                        
                        # Add to results
                        config_results.append(result)
                        merged_json.append(result)
                        
                        # Save intermediate results if needed
                        if config.save and (i + 1) % config.save == 0:
                            # Save config-specific files
                            config.save_json(os.path.join(output_dir, f"{key}.json"), config_results)
                            #pd.DataFrame(config_results).to_excel(os.path.join(output_dir, f"{config.id}_{config.model}_{i+1}.xlsx"), index=False)
                            
                            # Save merged files
                            pd.DataFrame(merged_json).to_excel(os.path.join(output_dir, "all_results.xlsx"), index=False)
                            with open(merged_json_path, 'w', encoding='utf-8') as f:
                                json.dump(merged_json, f, indent=2, ensure_ascii=False)
                            
                            logger.info(f"Saved intermediate results at {key}")
                        
                        # Update checkpoint
                        checkpoints[key] = 1
                        
                    except Exception as e:
                        logger.error(f"Error processing {key}: {e}")
                        checkpoints[key] = 0  # Mark as failed
                        result = {
                            "iteration": i+1,
                            "error": str(e),
                            "model": config.model,
                            "timestamp": datetime.now().isoformat(),
                            "id": config.id
                        }
                        config_results.append(result)
                        merged_json.append(result)
                    
                    # Save current state periodically
                    with open(checkpoint_path, 'w', encoding='utf-8') as f:
                        json.dump(checkpoints, f, indent=2)
                
                if processed:
                    # Save final config results
                    if config_results:
                        config.save_json(os.path.join(output_dir, f"{config.id}_{config.model}.json"), config_results)
                        pd.DataFrame(config_results).to_excel(os.path.join(output_dir, f"{config.id}_{config.model}.xlsx"), index=False)
            
            except Exception as e:
                logger.error(f"Critical error in configuration {idx+1}: {e}")
                continue
            
            # Final merged save
            if merged_json:
                pd.DataFrame(merged_json).to_excel(os.path.join(output_dir, "all_results.xlsx"), index=False)
                with open(merged_json_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_json, f, indent=2)
                logger.info("Successfully updated merged results")
                
    except Exception as e:
        logger.error(f"Critical error in experiment: {e}")


# Example usage
if __name__ == '__main__':
    
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