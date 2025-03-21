# from vertexai.generative_models import GenerationResponse
from openai import ChatCompletion
from src.llm.strategy import GenerationStrategyFactory
from src.llm.factory import ModelFactoryProvider
from src.config.logging import logger
from src.config.setup import *
from src.utils.io import extract_json_from_response
from typing import Optional
from typing import List 
from typing import Dict 
from typing import Any 
import time

class ResponseGenerator:
    """
    Handles response generation using various generation strategies and model factories. 
    The `strategy_type` in the configuration specifies the generation strategy.

    Attributes:
        model_factory: Instance of ModelFactoryProvider for creating model instances.
        generation_strategy: Strategy selected for content generation.
    """

    def __init__(self, llm_type: str = "openai", strategy_type: str = "default") -> None:
        """
        Initializes ResponseGenerator with the specified strategy type.

        Args:
            strategy_type (str): Type of strategy for content generation. Defaults to "default".
        """
        self.model_factory = ModelFactoryProvider.get_instance(llm_type)
        self.generation_strategy = GenerationStrategyFactory.get_strategy(strategy_type)

    def generate_response(self, model_name: str, system_instruction: str, contents: List[str], response_schema: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> ChatCompletion:
        """
        Generates a response based on the provided model name, system instruction, contents, response schema, and tools.

        Args:
            model_name (str): Name of the model for generation.
            system_instruction (str): Instruction or prompt for the model.
            contents (List[str]): Content input list for response generation.
            response_schema (Optional[Dict[str, Any]]): Schema defining response structure and constraints (default is None).
            tools (List[Any]): Tools passed to the model for content generation (default is None).

        Returns:
            ChatCompletion: Generated response object.

        Raises:
            Exception: If an error occurs during response generation.
        """
        logger.info("Starting response generation.")
        
        try:
            logger.info(f"Creating model instance for: {model_name}")
            model = self.model_factory
            logger.info("Model created successfully.")
            
            # Prepare generation configuration and safety settings
            generation_config = self.generation_strategy.create_generation_config(response_schema)
            safety_settings = self.generation_strategy.create_safety_settings()
            messages = [{"role": "system", "content": system_instruction}]
            messages.extend([{"role": "user", "content": content} for content in contents])

            kwargs = {"model": model_name, "messages": messages, **generation_config, **safety_settings}
            
            response = model.complete(**kwargs)
            response = model.parse_response(response)
            if response_schema:
                response = extract_json_from_response(response)
            logger.info("Response generated successfully.")
            return response 
        
        except Exception as e:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    logger.error(f"Error generating response, attempt {retry_count + 1} of {max_retries}: {e}")
                    time.sleep(5)  # Wait between retries
                    return self._retry_generate_response(model, kwargs)
                except Exception as retry_error:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Max retries ({max_retries}) reached. Final error: {retry_error}")
                        raise
                    e = retry_error

    def _retry_generate_response(self, model, kwargs, response_schema: Optional[Dict[str, Any]] = None) -> ChatCompletion:
        """
        Retries response generation once after a 429 quota limit error.

        Args:
            model: Model instance to retry with.
            contents (List[str]): Content input for response generation.
            kwargs (Dict[str, Any]): Keyword arguments for the model.

        Returns:
            GenerationResponse: The response after retry.
            
        Raises:
            Exception: If the retry fails or another error occurs.
        """
        try:
            response = model.complete(**kwargs)
            response = model.parse_response(response)
            if response_schema:
                response = extract_json_from_response(response)
            logger.info("Response generated successfully after retry.")
            return response
        except Exception as retry_error:
            logger.error(f"Retry failed: {retry_error}")
            raise
