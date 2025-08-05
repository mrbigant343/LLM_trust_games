from __future__ import annotations
from dotenv import load_dotenv
from openai import OpenAI
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from yandex_cloud_ml_sdk import YCloudML
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

# Read all models from models.txt
try:
    with open('app/models.txt', 'r') as f:
        models_list = [line.strip() for line in f if not line.strip().startswith('#')]
except Exception as e:
    logger.error(f"Error reading models.txt: {e}")
    models_list = ['meta-llama/llama-3.3-8b-instruct:free', 'qwen/qwen3-30b-a3b:free']

def _generate_single_prompt(prompt) -> str:
        '''
        Internal method for a single model call.
        Handles dynamic parameters (e.g. substitutes sum into prompt).
        '''
        
        # Replace {sum} placeholder with actual sum value
        #if Sum and prompt:
            #prompt = prompt.replace("{sum}", str(Sum))
        
        # Add JSON instruction to the prompt
        json_instruction = """
        Please respond in the following JSON format:
        {
            "decision": <your_decision_amount_as_integer>,
            "reasoning": "<your_reasoning_for_this_decision>"
        }
        IMPORTANT: Respond ONLY with valid JSON. Do not include any other text before or after the JSON.
        """.strip()
        full_prompt = f"{prompt}\n\n{json_instruction}"
        
        return full_prompt

class ExcelLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.data: pd.DataFrame = None   # Raw data from Excel
        self.configs: List['LLMClientConfig'] = []  # Ready-made configs
    

    def load(self) -> List['LLMClientConfig']:
        '''Loads Excel, converts strings to LLMClientConfig.
        The Excel file must optionally contain the following columns:
        id (str) - id for string;
        system_prompt (str) - system prompt with character description;
        prompt (str) - game instructions;
        sum (int) - amount added to the instructions;
        n_requests (int) - number of iterations in which the model works with one prompt;
        temp (str) - temperature float (from 0 to 1)
        save (int) - step for saving results
        comment(str) - comment for researcher
        '''
        try:
            # Load Excel file
            self.data = pd.read_excel(self.filename)
            logger.info(f"Loaded Excel file: {self.filename}")
            
            # Convert each row to LLMClientConfig
            for index, row in self.data.iterrows():
                # Get model name from row if available, otherwise default
                row_model = row.get('model', None)
                
                # If no specific model in row, use all available models
                models_to_run = [row_model] if row_model and pd.notna(row_model) else models_list
                
                for model in models_to_run:
                    try:
                        config = LLMClientConfig(
                            id=row.get('id', None),
                            system_prompt=_generate_single_prompt(row.get('system_prompt', None)),
                            sum=int(row.get('sum', 0)) if pd.notna(row.get('sum')) else 0,
                            prompt= row.get('prompt', None),
                            model=model,  # Use the specific model
                            n_requests=int(row.get('n_requests', 1)) if pd.notna(row.get('n_requests')) else 1,
                            temp=float(row.get('temp', 0.7)) if pd.notna(row.get('temp')) else 0.7,
                            save=int(row.get('save')) if pd.notna(row.get('save')) else None,
                            comment=row.get('comment', None)
                        )
                        
                        # Validate config before adding
                        if config.validate_config():
                            self.configs.append(config)
                        else:
                            logger.warning(f"Invalid config for {model} at row {index}, skipping...")
                    except Exception as e:
                        logger.error(f"Error creating config for {model} at row {index}: {e}")
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
    id: Optional[str] = None            # Configuration ID
    model: str = models_list[0]   # Default to first model
    system_prompt: Optional[str] = None  # May be absent
    prompt: Optional[str] = None        # May be absent
    sum: Optional[int] = 0              # Default value is 0
    n_requests: int = 1                  # Default value is 1
    temp: float = 0.7                   # Default value is 0.7
    save: Optional[int] = None          # Save step (optional)
    comment: Optional[str] = None       # Researcher comment

    def init_openrouter(self) -> OpenAI:
        '''
        Connects to OpenRouter API for the current model.
        Uses parameters model, system_prompt, temp.
        Additionally, API key is passed.
        '''
        token = os.getenv('OPENROUTER_API_KEY')
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
    
    def _run_openrouter(self):
        client = self.init_openrouter()

        messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.prompt}
                ]
        def _completion_call():
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temp,
                #max_tokens=1000
            )

        response = LLMClientConfig._retry_on_failure(_completion_call)
        return response.choices[0].message.content

    def init_gigachat(self) -> GigaChat:
        '''
        Connects to Gigachat API.
        Uses parameters model, system_prompt, temp.
        Additionally, API key is passed.
        '''
        token = os.getenv('GIGACHAT_API_KEY')
        try:
            client = GigaChat(
                credentials=token,
                model='GigaChat-2-Max',  # Use the specific Gigachat model
                verify_ssl_certs=False
            )
            logger.info(f"Initialized Gigachat client for model: {self.model}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Gigachat: {e}")
            raise

    def _run_gigachat(self) -> str:
        
        client = self.init_gigachat()
        # Create chat object
        chat = Chat(
                    messages=[
                        Messages(role=MessagesRole.SYSTEM, content=self.system_prompt or ""),
                        Messages(role=MessagesRole.USER, content=self.prompt)
                    ]
                )
                
                # Make API call
        def _chat_call():
            return client.chat(chat)

        response = LLMClientConfig._retry_on_failure(_chat_call)
        return response.choices[0].message.content
        

    def init_yandexgpt(self) -> YCloudML:
        '''
        Connects to Yandex Cloud API.
        Requires YANDEX_FOLDER_ID environment variable and token as IAM token.
        '''
        try:
            # Get Yandex Cloud folder ID from environment
            folder_id = os.getenv("YANDEX_FOLDER_ID")
            token = os.getenv("YANDEX_API_KEY")
            if not folder_id:
                raise ValueError("YANDEX_FOLDER_ID environment variable is not set")
            
            # Initialize client
            client = YCloudML(
                folder_id=folder_id,
                auth=token,  # token should be IAM token
            )
            logger.info(f"Initialized YandexGPT client for model: {self.model}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize YandexGPT: {e}")
            raise
    
    def _run_yandexgpt(self):
        client = self.init_yandexgpt()
        messages = [
                    {"role": "system", "text": self.system_prompt or ""},
                    {"role": "user", "text": self.prompt}
                ]
                
                # Configure and run model
        def _completion_call():
            return (client.models.completions("yandexgpt")
                    .configure(temperature=self.temp)
                    .run(messages))

        response = LLMClientConfig._retry_on_failure(_completion_call)
        return response[0].text

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
    
    def parse_json(self, output: str) -> dict:
        '''
        Converts model results to json format.
        Adds timestamp and tries to extract numbers.
        '''
        pattern = r'<think>[\s\S]*?</think>'
        output = re.sub(pattern, '', output)
        result = {
            "id": self.id,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "prompt": self.prompt, 
            "sum": self.sum,             
            "n_requests": self.n_requests,              
            "temp": self.temp,              
            "save": self.save,    
            "comment": self.comment,
            "response": output,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to parse as JSON
        cleaned_output = output.strip('````pythonjson\n "')
        s = cleaned_output.find('{')
        e = cleaned_output.rfind('}')
        if  s != -1 and e != -1:
            cleaned_output = cleaned_output[s:e+1]
        # Step 2: Try parsing from direct JSON fragment (best case)
        #if cleaned_output.startswith('{') and cleaned_output.endswith('}'):
        try:
            parsed_json = json.loads(cleaned_output)
            result.update(parsed_json)
            logger.debug(f"Successfully parsed JSON for model {self.model}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed for model {self.model}. Response: {output[:100]}...")
            # Re-raise the JSONDecodeError so it can be caught by retry mechanism
            raise json.JSONDecodeError
        return result
    
    def check_and_run_model(self):
        if "gigachat" in self.model.lower():    
            return self._run_gigachat()
        elif "yandexgpt" in self.model.lower():
            return self._run_yandexgpt()
        return self._run_openrouter()

    def save_json(self, filename: str, data: List[Dict[str, Any]] = None) -> None:
        try:
            if data is None:
                data = []

            # Add model name to filename if not present
            if self.model not in filename:
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{self.model}{ext}"

            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON results to: {filename}")
        except Exception as e:
            logger.error(f"Error saving JSON file: {e}")

    def save_excel(self, filename: str, data: List[Dict[str, Any]]) -> None:
        '''
        Saves a complete .xlsx file with all iteration results.
        Include the model name in the filename for clarity.
        '''
        try:
            if self.model not in filename:
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{self.model}{ext}"
            
            df = pd.DataFrame(data)
            df.to_excel(filename, index=False)
            logger.info(f"Saved Excel results to: {filename}")
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")


    @staticmethod
    def _handle_api_error(error: Exception) -> str:
        '''
        Internal method for detecting model run errors.
        '''
        
        error_msg = str(error)
        if "rate limit" in error_msg.lower():
            return "rate_limit"
        elif "timeout" in error_msg.lower():
            return "timeout"
        elif "authentication" in error_msg.lower():
            return "auth_error"
        elif "JSON" in error_msg.lower():
            return "parsing_error"
        else:
            return "unknown_error"

    @staticmethod
    def _retry_on_failure(func, max_retries: int = 3):
        '''
        Function to retry a model call when an error occurs.
        '''
        logger.debug(f"Retrying function: {func}, type: {type(func)}")
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
