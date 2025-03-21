# from vertexai.generative_models import GenerativeModel
from openai import ChatCompletion
from openai import OpenAI
from src.config.logging import logger
from abc import abstractmethod
from typing import Optional, Any
from abc import ABC
import os
import dotenv
from typing import Union

dotenv.load_dotenv()


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, *args, **kwargs) -> Any:
        """Standardized completion method across different LLM providers"""
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> Any:
        """Standardized response parsing across different LLM providers"""
        pass

class OpenAIClient(BaseLLMClient):
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self, *args, **kwargs):
        return self._client.chat.completions.create(*args, **kwargs)

    def parse_response(self, response):
        # OpenAI specific response parsing
        return response.choices[0].message.content

class DeepSeekClient(BaseLLMClient):
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/beta")

    def complete(self, *args, **kwargs):
        return self._client.chat.completions.create(*args, **kwargs)

    def parse_response(self, response):
        # DeepSeek specific response parsing
        return response.choices[0].message.content

# class AnthropicClient(BaseLLMClient):
#     def __init__(self, client):
#         self._client = client

#     def complete(self, *args, **kwargs):
#         return self._client.messages.create(*args, **kwargs)

#     def parse_response(self, response):
#         # Anthropic specific response parsing
#         return response.choices[0].message.content

# class OpenAIClient(ModelFactory):
#     """
#     Concrete implementation of the ModelFactory for OpenAI models.
#     """
#     def __init__(self):
#         self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     def create_model(self) -> ChatCompletion:
#         """
#         Creates and returns an instance of a OpenAI GenerativeModel.
#         """
#         try:
#             self.client.complete = self.client.chat.completions.create
#             self.client.parse_response = self.parse_response
#             return self.client
#         except Exception as e:
#             logger.error(f"Error creating OpenAI Model: {e}")
#             raise
#     def create_embeddings(self, texts: Union[list[str], str]) -> list[list[float]]:
#         """
#         Create embeddings for a list of texts using OpenAI's API directly
#         """
#         response = self.client.embeddings.create(
#             model="text-embedding-3-small",
#             input=texts,
#             dimensions=1536
#         )
#         return [item.embedding for item in response.data]
    
#     def parse_response(self, response: ChatCompletion) -> str:
#         """
#         Parses the response from OpenAI and returns the content
#         """
#         return response.choices[0].message.content

# class DeepSeekModelFactory(ModelFactory):
#     """
#     Concrete implementation of the ModelFactory for DeepSeek models.
#     """
#     def __init__(self):
#         self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

#     def create_model(self) -> ChatCompletion:
#         """
#         Creates and returns an instance of a DeepSeek GenerativeModel.
#         """
#         try:
#             self.client.complete = self.client.chat.completions.create
#             return self.client
#         except Exception as e:
#             logger.error(f"Error creating DeepSeek Model: {e}")
#             raise
#     def create_embeddings(self, texts: Union[list[str], str]) -> list[list[float]]:
#         """
#         Create embeddings for a list of texts using OpenAI's API directly
#         """
#         openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         response = openai_client.embeddings.create(
#             model="text-embedding-3-small",
#             input=texts,
#             dimensions=1536
#         )
#         return [item.embedding for item in response.data]
    
#     def parse_response(self, response: ChatCompletion) -> str:
#         """
#         Parses the response from OpenAI and returns the content
#         """
#         return response.choices[0].message.content
    


class ModelFactoryProvider:
    """
    Singleton provider for the ModelFactory.

    This class ensures that only one instance of the ModelFactory is created,
    providing a global access point for it.
    """
    _instance: Optional[BaseLLMClient] = None

    @staticmethod
    def get_instance(llm_type: str = "openai") -> BaseLLMClient:
        """
        Returns the singleton instance of the ModelFactory.

        If no instance exists, it creates one.

        Returns:
        --------
        ModelFactory: The singleton instance of the ModelFactory.
        """
        if ModelFactoryProvider._instance is None:
            ModelFactoryProvider._instance = OpenAIClient() if llm_type == "openai" else DeepSeekClient()
        else:
            if ModelFactoryProvider._instance.__class__ == OpenAIClient and llm_type == "deepseek":
                del ModelFactoryProvider._instance
                logger.warning("Switching to DeepSeek model")
                ModelFactoryProvider._instance = DeepSeekClient()
            elif ModelFactoryProvider._instance.__class__ == DeepSeekClient and llm_type == "openai":
                del ModelFactoryProvider._instance
                logger.warning("Switching to OpenAI model")
                ModelFactoryProvider._instance = OpenAIClient()
        return ModelFactoryProvider._instance
