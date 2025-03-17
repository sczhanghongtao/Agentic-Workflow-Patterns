# from vertexai.generative_models import GenerativeModel
from openai import ChatCompletion
from openai import OpenAI
from src.config.logging import logger
from abc import abstractmethod
from typing import Optional
from abc import ABC
import os
import dotenv

dotenv.load_dotenv()


class ModelFactory(ABC):
    """
    Abstract base class for creating generative models.

    This class defines the interface for creating generative models, 
    ensuring that subclasses implement the `create_model` method.
    """

    @abstractmethod
    def create_model(self, model_name: str, system_instruction: str):
        """
        Creates and returns an instance of a GenerativeModel.

        Args:
            model_name (str): The name of the model to create.
            system_instruction (str): The system instruction to initialize the model with.

        Returns:
        --------
        GenerativeModel: An instance of the GenerativeModel.

        Raises:
        -------
        NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the `create_model` method")

class OpenAIModelFactory(ModelFactory):
    """
    Concrete implementation of the ModelFactory for OpenAI models.
    """
    def create_model(self) -> ChatCompletion:
        """
        Creates and returns an instance of a OpenAI GenerativeModel.
        """
        try:
            model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model.complete = model.chat.completions.create
            return model
        except Exception as e:
            logger.error(f"Error creating OpenAI Model: {e}")
            raise

class ModelFactoryProvider:
    """
    Singleton provider for the ModelFactory.

    This class ensures that only one instance of the ModelFactory is created,
    providing a global access point for it.
    """
    _instance: Optional[ModelFactory] = None

    @staticmethod
    def get_instance() -> ModelFactory:
        """
        Returns the singleton instance of the ModelFactory.

        If no instance exists, it creates one.

        Returns:
        --------
        ModelFactory: The singleton instance of the ModelFactory.
        """
        if ModelFactoryProvider._instance is None:
            ModelFactoryProvider._instance = OpenAIModelFactory()
        return ModelFactoryProvider._instance
