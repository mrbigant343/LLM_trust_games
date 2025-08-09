# LLM Trust Games Experiment Framework

This project implements a framework for conducting trust game experiments with Large Language Models (LLMs). It allows researchers to systematically test how different LLMs behave in economic trust games by configuring prompts and collecting structured responses.

## Features

- **Multi-LLM Support**: Works with OpenRouter, GigaChat, and YandexGPT models
- **Configurable Experiments**: Define trust game scenarios via Excel configuration
- **Retry Mechanism**: Automatic retry with exponential backoff for API failures
- **Checkpointing**: Save progress and resume interrupted experiments
- **Structured Output**: Collect results in both JSON and Excel formats
- **Error Handling**: Comprehensive error handling and logging

## Project Structure

```
LLM_trust_games/
├── app/                 # Core framework modules
│   ├── models.txt       # List of supported models
│   └── model_framework.py # Main framework implementation
├── config.xlsx          # Experiment configuration
├── main.py              # Main experiment runner
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not in repo)
├── results/             # Experiment results
└── trust_game_env/      # Python virtual environment
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mrbigant343/LLM_trust_games.git
   cd LLM_trust_games
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv trust_game_env
   trust_game_env\Scripts\activate  #On Linux:  source trust_game_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   Create a `.env` file with your API keys:
   ```bash
   # OpenRouter API (required for most models)
   OPENROUTER_API_KEY=your_openrouter_api_key
   
   # GigaChat API (optional)
   GIGACHAT_API_KEY=your_gigachat_api_key
   
   # Yandex Cloud API (optional)
   YANDEX_API_KEY=your_yandex_api_key
   YANDEX_FOLDER_ID=your_yandex_folder_id
   ```

## Configuration

### Excel Configuration File

The `config.xlsx` file defines experiment parameters:

| Column | Description | Required |
|--------|-------------|----------|
| id | Unique identifier for the experiment | Yes |
| system_prompt | System prompt for the LLM | Yes |
| prompt | Game scenario prompt | Yes |
| n_requests | Number of iterations per model | No (default: 1) |
| temp | Temperature setting (0.0-1.0) | No (default: 0.7) |
| save | Save results every N iterations | No |
| comment | Research notes | No |

### Models Configuration

Edit `app/models.txt` to specify which models to use:
```
# Commented models are disabled
#deepseek/deepseek-chat-v3-0324
openai/o4-mini
#anthropic/claude-sonnet-4
```

## Usage

Run the trust game experiment:
```bash
python main.py
```

Results will be saved to the `results/` directory:
- Individual model results as JSON and Excel files
- Merged results from all experiments
- Checkpoint files for resuming interrupted experiments

## Experiment Process

1. **Configuration Loading**: Reads experiment parameters from `config.xlsx`
2. **Model Initialization**: Sets up connections to configured LLM APIs
3. **Prompt Generation**: Creates prompts according to the trust game framework
4. **Model Execution**: Sends prompts to LLMs with retry logic
5. **Response Parsing**: Extracts structured data from model responses
6. **Result Saving**: Stores results in JSON and Excel formats
7. **Checkpointing**: Maintains progress tracking

## Output Format

Results are saved in structured JSON format:
```json
{
  "id": "experiment_id",
  "model": "model_name",
  "system_prompt": "System prompt text",
  "prompt": "Game prompt text",
  "response": "Model response text",
  "decision": 50,
  "reasoning": "Model's reasoning for the decision",
  "timestamp": "ISO timestamp"
}
```

## Adding New Models

To add support for new models:

1. Update the model initialization in `app/model_framework.py`
2. Add the model name to `app/models.txt`
3. Ensure proper API configuration in `.env`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue on GitHub.
