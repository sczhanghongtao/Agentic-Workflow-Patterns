import pytest
import re
import json
from unittest.mock import Mock, patch
from src.llm.generate import ResponseGenerator
from src.llm.factory import ModelFactoryProvider, OpenAIModelFactory
from src.llm.strategy import DefaultGenerationStrategy, GenerationStrategyFactory

def parse_json_output(string):
    # Use a regular expression to find the JSON part in the string
    json_match = re.search(r'```json\n(.*?)\n\s*```', string, re.DOTALL)
    
    if not json_match:
        parsed_json = json.loads(string)
    else:
    
        json_str = json_match.group(1).strip()
    
        # Parse the JSON string
        parsed_json = json.loads(json_str)
    
    return parsed_json

# Test data
MOCK_MODEL_NAME = "gpt-4o-mini"
MOCK_SYSTEM_INSTRUCTION = "You are a helpful assistant."
MOCK_CONTENTS = ["Hello, how are you? return your response in json object with the key 'answer'"]
MOCK_RESPONSE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "response", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}}}

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    mock_client.complete = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock()
    return mock_client

@pytest.fixture
def response_generator():
    """Create ResponseGenerator instance for testing"""
    return ResponseGenerator()

class TestResponseGenerator:
    def test_init_default_strategy(self):
        """Test ResponseGenerator initialization with default strategy"""
        generator = ResponseGenerator()
        assert isinstance(generator.generation_strategy, DefaultGenerationStrategy)
        assert generator.model_factory is not None

    @patch('src.llm.factory.OpenAI')
    def test_generate_response_success(self, mock_openai, response_generator, mock_openai_client):
        """Test successful response generation"""
        mock_openai.return_value = mock_openai_client
        mock_response = Mock()
        mock_openai_client.chat.completions.create.return_value = mock_response

        response = response_generator.generate_response(
            MOCK_MODEL_NAME,
            MOCK_SYSTEM_INSTRUCTION,
            MOCK_CONTENTS,
            MOCK_RESPONSE_SCHEMA
        )

        assert response == mock_response
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch('src.llm.factory.OpenAI')
    def test_generate_response_retry_on_error(self, mock_openai, response_generator, mock_openai_client):
        """Test response generation with retry on error"""
        mock_openai.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("Rate limit exceeded"),
            Mock()  # Successful response on retry
        ]

        response = response_generator.generate_response(
            MOCK_MODEL_NAME,
            MOCK_SYSTEM_INSTRUCTION,
            MOCK_CONTENTS
        )

        assert mock_openai_client.chat.completions.create.call_count == 2

    def test_generate_response(self, response_generator):
        response = response_generator.generate_response(
            MOCK_MODEL_NAME,
            MOCK_SYSTEM_INSTRUCTION,
            MOCK_CONTENTS
        )
        response_json = parse_json_output(response.choices[0].message.content)

        assert "answer" in response_json
        assert isinstance(response_json["answer"], str)

class TestModelFactory:
    def test_model_factory_singleton(self):
        """Test ModelFactoryProvider singleton pattern"""
        factory1 = ModelFactoryProvider.get_instance()
        factory2 = ModelFactoryProvider.get_instance()
        assert factory1 is factory2
        assert isinstance(factory1, OpenAIModelFactory)

    @patch('src.llm.factory.OpenAI')
    def test_create_model(self, mock_openai, mock_openai_client):
        """Test OpenAIModelFactory create_model method"""
        mock_openai.return_value = mock_openai_client
        factory = OpenAIModelFactory()
        model = factory.create_model()
        assert model == mock_openai_client
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
