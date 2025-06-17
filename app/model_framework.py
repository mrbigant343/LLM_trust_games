from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
import os
import time
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openrouter_models = ['meta-llama/llama-3.3-8b-instruct:free', 'qwen/qwen3-30b-a3b:free']

class ExcelLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.data: pd.DataFrame = None   # Raw data from Excel
        self.configs: List['LLMClientConfig'] = []  # Ready-made configs

    def load(self) -> List['LLMClientConfig']:
        '''Loads Excel, converts strings to LLMClientConfig.
        The Excel file must optionally contain the following columns:
        system_prompt (str) - system prompt with character description;
        prompt (str) - game instructions;
        sum (int) - amount added to the instructions;
        n_requests (int) - number of iterations in which the model works with one prompt;
        temp (str) - temperature float (from 0 to 1)
        save (int) - step for saving results
        '''
        try:
            # Load Excel file
            self.data = pd.read_excel(self.filename)
            logger.info(f"Loaded Excel file: {self.filename}")
            
            # Convert each row to LLMClientConfig
            for index, row in self.data.iterrows():
                config = LLMClientConfig(
                    system_prompt=row.get('system_prompt', None),
                    prompt=row.get('prompt', None),
                    sum=int(row.get('sum', 0)) if pd.notna(row.get('sum')) else 0,
                    model=row.get('model', openrouter_models[0]),  # Default to first model
                    n_requests=int(row.get('n_requests', 1)) if pd.notna(row.get('n_requests')) else 1,
                    temp=float(row.get('temp', 0.7)) if pd.notna(row.get('temp')) else 0.7,
                    save=int(row.get('save')) if pd.notna(row.get('save')) else None
                )
                
                # Validate config before adding
                if config.validate_config():
                    self.configs.append(config)
                else:
                    logger.warning(f"Invalid config at row {index}, skipping...")
                    
            logger.info(f"Successfully loaded {len(self.configs)} valid configurations")
            return self.configs
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            return []

@dataclass
class LLMClientConfig:
    '''
    Main class for LLM initialization
    '''
    model: str = openrouter_models[0]    # Default to first model
    system_prompt: Optional[str] = None  # May be absent
    prompt: Optional[str] = None        # May be absent
    sum: Optional[int] = 0              # Default value is 0
    n_requests: int = 1                  # Default value is 1
    temp: float = 0.7                   # Default value is 0.7
    save: Optional[int] = None          # Save step (optional)


    def init_openrouter(self, token: str) -> OpenAI:
        '''
        Connects to OpenRouter API.
        Uses parameters model, system_prompt, temp.
        Additionally, API key is passed
        '''
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=token,
            )
            logger.info(f"Initialized OpenRouter client for model: {self.model}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter: {e}")
            raise

    def init_gigachat(self, token: str) -> str:
        '''
        Connects to Gigachat API.
        Uses parameters model, system_prompt, temp.
        Additionally, API key is passed
        '''
        # Placeholder - to be implemented when needed
        logger.info("Gigachat initialization - placeholder (inactive)")
        return "gigachat_client_placeholder"

    def init_yandexgpt(self, token: str) -> str:
        '''
        Connects to Yandex Cloud API.
        Uses parameters model, system_prompt, temp.
        Additionally, an API key or IAM token is passed.
        '''
        # Placeholder - to be implemented when needed
        logger.info("YandexGPT initialization - placeholder (inactive)")
        return "yandexgpt_client_placeholder"

    def _generate_json_instruction(self) -> str:
        '''
        Generates JSON format instructions for the model response.
        This tells the model exactly how to format its response.
        '''
        json_instruction = """
Please respond in the following JSON format:
{
    "decision": <your_decision_amount_as_integer>,
    "reasoning": "<your_reasoning_for_this_decision>",
    "kept_amount": <amount_you_keep_as_integer>,
    "given_amount": <amount_you_give_as_integer>,
    "confidence": <confidence_level_from_1_to_10>
}

Example:
{
    "decision": 30,
    "reasoning": "I want to be fair and share some of the money while keeping most for myself",
    "kept_amount": 70,
    "given_amount": 30,
    "confidence": 8
}

IMPORTANT: Respond ONLY with valid JSON. Do not include any other text before or after the JSON.
"""
        return json_instruction.strip()

    def _generate_single_prompt(self) -> str:
        '''
        Internal method for a single model call.
        Handles dynamic parameters (e.g. substitutes sum into prompt).
        '''
        # Create the full prompt by substituting variables
        full_prompt = self.prompt
        
        # Replace {sum} placeholder with actual sum value
        if self.sum and full_prompt:
            full_prompt = full_prompt.replace("{sum}", str(self.sum))
        
        # Add JSON instruction to the prompt
        json_instruction = self._generate_json_instruction()
        full_prompt = f"{full_prompt}\n\n{json_instruction}"
        
        return full_prompt

    def validate_config(self) -> bool:
        '''
        Checks that all required fields (eg model) are filled in and temp is correct.
        '''
        # Check required fields
        if not self.model:
            logger.error("Model field is required")
            return False
        
        # Check temperature range
        if not (0.0 <= self.temp <= 1.0):
            logger.error(f"Temperature must be between 0 and 1, got: {self.temp}")
            return False
        
        # Check n_requests is positive
        if self.n_requests <= 0:
            logger.error(f"n_requests must be positive, got: {self.n_requests}")
            return False
        
        return True

    def run_model(self, token: str) -> List[Dict[str, Any]]:
        '''
        Function for generating model responses. In it, init is run.
        Takes all model parameters and starts the run and save cycle.
        The function should handle errors and process until the number of responses equals n_requests
        '''
        if not self.validate_config():
            raise ValueError("Invalid configuration")
        
        # Initialize client (OpenRouter only for now)
        client = self.init_openrouter(token)
        results = []
        
        for i in range(self.n_requests):
            try:
                logger.info(f"Processing request {i+1}/{self.n_requests}")
                
                # Generate prompt with substitutions
                full_prompt = self._generate_single_prompt()
                
                # Prepare messages
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                if full_prompt:
                    messages.append({"role": "user", "content": full_prompt})
                
                # Make API call with retry logic
                response = self._retry_on_failure(
                    lambda: client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temp,
                        max_tokens=1000
                    ),
                    max_retries=3
                )
                
                # Parse response
                if response and response.choices:
                    response_text = response.choices[0].message.content
                    result = self.parse_json(response_text)
                    result.update({
                        "iteration": i + 1,
                        "model": self.model,
                        "temperature": self.temp,
                        "sum": self.sum
                    })
                    results.append(result)
                    
                    # Save intermediate results if save step is specified
                    if self.save and (i + 1) % self.save == 0:
                        self.save_json(f"intermediate_results_{i+1}.json", results)
                
            except Exception as e:
                logger.error(f"Error in request {i+1}: {e}")
                # Continue with next request rather than failing completely
                results.append({
                    "iteration": i + 1,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model
                })
        
        return results

    def parse_json(self, output: str) -> dict:
        '''
        Converts model results to json format.
        Adds timestamp
        '''
        result = {
            "response": output,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to extract numeric values from response (for trust game decisions)
        # Look for numbers that might represent monetary decisions
        numbers = re.findall(r'\b\d+\b', output)
        if numbers:
            result["extracted_numbers"] = [int(n) for n in numbers]
            result["decision"] = int(numbers[0]) if numbers else None
        
        # Try to parse as JSON if the response looks like JSON
        output_stripped = output.strip()
        if output_stripped.startswith('{') and output_stripped.endswith('}'):
            try:
                parsed_json = json.loads(output_stripped)
                result.update(parsed_json)
            except json.JSONDecodeError:
                logger.warning("Response looks like JSON but failed to parse")
        
        return result

    def save_json(self, filename: str, data: List[Dict[str, Any]] = None) -> None:
        '''
        Saves a temporary json file with the iteration results
        '''
        try:
            if data is None:
                data = []
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON results to: {filename}")
        except Exception as e:
            logger.error(f"Error saving JSON file: {e}")

    def save_excel(self, filename: str, data: List[Dict[str, Any]]) -> None:
        '''
        Saves a complete .xlsx file with all iteration results
        '''
        try:
            df = pd.DataFrame(data)
            df.to_excel(filename, index=False)
            logger.info(f"Saved Excel results to: {filename}")
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")

    @staticmethod
    def _handle_api_error(error: Exception) -> str:
        '''
        Internal method for detecting model run errors
        '''
        error_msg = str(error)
        if "rate limit" in error_msg.lower():
            return "rate_limit"
        elif "timeout" in error_msg.lower():
            return "timeout"
        elif "authentication" in error_msg.lower():
            return "auth_error"
        else:
            return "unknown_error"

    @staticmethod
    def _retry_on_failure(func, max_retries: int = 3):
        '''
        Function to call the run_model() method again when an error occurs
        '''
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                error_type = LLMClientConfig._handle_api_error(e)
                wait_time = 2 ** attempt  # Exponential backoff
                
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                    raise e
                
                logger.warning(f"Attempt {attempt + 1} failed ({error_type}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        return None


