import pytest
import re
import json
from unittest.mock import Mock, patch
from src.llm.generate import ResponseGenerator
from src.llm.factory import ModelFactoryProvider, OpenAIClient
from src.llm.strategy import DefaultGenerationStrategy, GenerationStrategyFactory
# Test data
MOCK_MODEL_NAME = "gpt-4o-mini"
MOCK_SYSTEM_INSTRUCTION = "You are a helpful assistant."
MOCK_CONTENTS = ["Hello, how are you? return your response in json object with the key 'answer'"]
MOCK_RESPONSE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "response", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}}}

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    mock_client._client = Mock()
    mock_client._client.chat.completions.create = Mock()
    
    # Explicitly define the forwarding behavior
    mock_client.complete.side_effect = lambda *args, **kwargs: mock_client._client.chat.completions.create(*args, **kwargs)
    
    mock_client.parse_response = Mock()
    return mock_client

@pytest.fixture
def response_generator(mock_openai_client):
    """Create ResponseGenerator instance for testing with mocked dependencies"""
    with patch('src.llm.factory.ModelFactoryProvider.get_instance') as mock_factory_provider:
        # Simply return the mock_openai_client directly
        mock_factory_provider.return_value = mock_openai_client
        response_generator = ResponseGenerator()
        return response_generator

class TestResponseGenerator:
    def test_init_default_strategy(self):
        """Test ResponseGenerator initialization with default strategy"""
        generator = ResponseGenerator()
        assert isinstance(generator.generation_strategy, DefaultGenerationStrategy)
        assert generator.model_factory is not None

    def test_generate_response_success(self, mock_openai_client, response_generator):
        """Test successful response generation"""
        # Set up the mock response
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content='{"answer": "Test response"}'))]
        expected_response = '{"answer": "Test response"}'
        # mock_openai_client.complete.return_value = expected_response
        mock_openai_client._client.chat.completions.create.return_value = mock_completion
        mock_openai_client.parse_response.return_value = expected_response

        response = response_generator.generate_response(
            MOCK_MODEL_NAME,
            MOCK_SYSTEM_INSTRUCTION,
            MOCK_CONTENTS,
            MOCK_RESPONSE_SCHEMA
        )
        
        # Assert the response is correct
        assert response == {"answer": "Test response"}
        
        # Verify the mock was called
        response_generator.model_factory.complete.assert_called_once()
        response_generator.model_factory._client.chat.completions.create.assert_called_once()

    @patch('src.llm.factory.OpenAIClient')
    def test_generate_response_retry_on_error(self, mock_openai, response_generator, mock_openai_client):
        """Test response generation with retry on error"""
        mock_openai.return_value = mock_openai_client
        mock_openai_client._client.chat.completions.create.side_effect = [
            Exception("Rate limit exceeded"),
            Mock()  # Successful response on retry
        ]

        response = response_generator.generate_response(
            MOCK_MODEL_NAME,
            MOCK_SYSTEM_INSTRUCTION,
            MOCK_CONTENTS,
            MOCK_RESPONSE_SCHEMA
        )

        assert mock_openai_client._client.chat.completions.create.call_count == 2

    def test_generate_response(self):
        response_generator = ResponseGenerator()
        response = response_generator.generate_response(
            MOCK_MODEL_NAME,
            MOCK_SYSTEM_INSTRUCTION,
            MOCK_CONTENTS,
            MOCK_RESPONSE_SCHEMA
        )

        assert "answer" in response
        assert isinstance(response["answer"], str)

class TestModelFactory:
    def test_model_factory_singleton(self):
        """Test ModelFactoryProvider singleton pattern"""
        factory1 = ModelFactoryProvider.get_instance("openai")
        factory2 = ModelFactoryProvider.get_instance("openai")
        assert factory1 is factory2
        assert isinstance(factory1, OpenAIClient)

    @patch('src.llm.factory.OpenAI')
    def test_create_model(self, mock_openai, mock_openai_client):
        """Test OpenAIClient create_model method"""
        mock_openai.return_value = mock_openai_client
        factory = OpenAIClient()
        model = factory
        assert model._client == mock_openai_client
        assert hasattr(model, 'complete')

class TestGenerationStrategy:
    def test_default_strategy_generation_config(self):
        """Test DefaultGenerationStrategy generation config creation"""
        strategy = DefaultGenerationStrategy()
        config = strategy.create_generation_config(MOCK_RESPONSE_SCHEMA)
        
        assert isinstance(config, dict)
        assert config['temperature'] == 1
        assert config['top_p'] == 1
        assert config['stream'] is False
        assert config['n'] == 1
        assert config['response_format'] == MOCK_RESPONSE_SCHEMA

    def test_default_strategy_safety_settings(self):
        """Test DefaultGenerationStrategy safety settings creation"""
        strategy = DefaultGenerationStrategy()
        settings = strategy.create_safety_settings()
        assert isinstance(settings, dict)
        assert len(settings) == 0  # Currently returns empty dict

    def test_strategy_factory(self):
        """Test GenerationStrategyFactory strategy creation"""
        strategy = GenerationStrategyFactory.get_strategy()
        assert isinstance(strategy, DefaultGenerationStrategy)

        with pytest.raises(ValueError):
            GenerationStrategyFactory.get_strategy("unknown_strategy")
