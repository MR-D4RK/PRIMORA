import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora.brain import Brain, AuroraBrain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig


def test_brain_interface():
    class MockBrain(Brain):
        def __init__(self, model: str):
            self._model = model

        def generate(self, messages: list[dict[str, str]]) -> BrainResult:
            return BrainResult.success("mock", self._model)

        def get_model_name(self) -> str:
            return self._model

    brain = MockBrain("test-model")
    result = brain.generate([{"role": "user", "content": "Hi"}])
    assert result.success
    assert result.content == "mock"
    assert brain.get_model_name() == "test-model"


def test_aurora_brain_requires_dependencies():
    with patch.dict('sys.modules', {'torch': None, 'transformers': None, 'peft': None}):
        brain = AuroraBrain()
        result = brain.generate([{"role": "user", "content": "Hi"}])
        assert not result.success
        assert "Missing dependencies" in result.error


def test_aurora_brain_missing_messages():
    brain = AuroraBrain()
    result = brain.generate("not a list")
    assert not result.success
    assert "Missing or invalid 'messages'" in result.error


def test_aurora_brain_get_model_name():
    config = AuroraConfig(
        model_path="/fake/path/model",
        adapter_path="/fake/path/adapter"
    )
    brain = AuroraBrain(config=config)
    assert brain.get_model_name() == "model"


def test_aurora_brain_initialization_error():
    config = AuroraConfig(
        model_path="/nonexistent/model",
        adapter_path="/nonexistent/adapter"
    )
    brain = AuroraBrain(config=config)
    result = brain.generate([{"role": "user", "content": "Hi"}])
    assert not result.success
    assert "Model initialization failed" in result.error or "Model path not found" in result.error


def test_aurora_config_from_env():
    os.environ["PRIMORA_AURORA_MODEL_PATH"] = "/custom/model"
    os.environ["PRIMORA_AURORA_ADAPTER_PATH"] = "/custom/adapter"
    os.environ["PRIMORA_AURORA_MAX_TOKENS"] = "512"
    os.environ["PRIMORA_AURORA_TEMPERATURE"] = "0.8"
    os.environ["PRIMORA_AURORA_TOP_P"] = "0.95"
    os.environ["PRIMORA_AURORA_REPETITION_PENALTY"] = "1.1"
    try:
        config = AuroraConfig.from_env()
        assert config.model_path == "/custom/model"
        assert config.adapter_path == "/custom/adapter"
        assert config.max_new_tokens == 512
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.repetition_penalty == 1.1
    finally:
        for key in ["PRIMORA_AURORA_MODEL_PATH", "PRIMORA_AURORA_ADAPTER_PATH", 
                    "PRIMORA_AURORA_MAX_TOKENS", "PRIMORA_AURORA_TEMPERATURE",
                    "PRIMORA_AURORA_TOP_P", "PRIMORA_AURORA_REPETITION_PENALTY"]:
            os.environ.pop(key, None)


def test_aurora_config_validate_missing():
    config = AuroraConfig(model_path="/nonexistent", adapter_path="/nonexistent")
    err = config.validate()
    assert err is not None
    assert "not found" in err


def test_aurora_brain_generation_logic():
    """Test the generation logic by mocking the internal imports"""
    import builtins
    original_import = builtins.__import__
    
    mock_torch = MagicMock()
    mock_transformers = MagicMock()
    mock_peft = MagicMock()
    
    def mock_import(name, *args, **kwargs):
        if name == 'torch':
            return mock_torch
        elif name == 'transformers':
            return mock_transformers
        elif name == 'peft':
            return mock_peft
        return original_import(name, *args, **kwargs)
    
    with patch('builtins.__import__', side_effect=mock_import):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        # Return a proper tensor-like object with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: mock_torch.tensor([[1, 2, 3]])
        mock_inputs.__setitem__ = lambda self, key, value: None
        mock_inputs.shape = [1, 3]
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode.return_value = "Generated response"
        mock_tokenizer.eos_token_id = 2
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_base_model = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_base_model

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [mock_torch.tensor([1, 2, 3, 4, 5, 6])]
        mock_peft.PeftModel.from_pretrained.return_value = mock_model

        config = AuroraConfig(model_path="/fake/model", adapter_path="/fake/adapter")
        brain = AuroraBrain(config=config)
        
        result = brain.generate([{"role": "user", "content": "Hello"}])
        
        assert result.success
        assert result.content == "Generated response"
        assert result.model == "model"
        
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 256
        assert call_kwargs["temperature"] == 0.75
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["repetition_penalty"] == 1.08
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["use_cache"] is True


if __name__ == "__main__":
    test_brain_interface()
    test_aurora_brain_requires_dependencies()
    test_aurora_brain_missing_messages()
    test_aurora_brain_get_model_name()
    test_aurora_brain_initialization_error()
    test_aurora_config_from_env()
    test_aurora_config_validate_missing()
    test_aurora_brain_generation_logic()
    print("All Brain and Config tests passed!")