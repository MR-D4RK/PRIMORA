import sys
import os

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora.brain import AuroraBrain
from PRIMORA.aurora.configuration import AuroraConfig


def test_aurora_brain_integration():
    config = AuroraConfig.from_env()
    err = config.validate()
    if err:
        print(f"SKIP: {err}")
        return
    
    print(f"Using model: {config.model_path}")
    print(f"Using adapter: {config.adapter_path}")

    brain = AuroraBrain(config=config)
    result = brain.generate([{"role": "user", "content": "Say hello briefly"}])

    if not result.success:
        print(f"FAIL: {result.error}")
        assert False, f"Brain generation failed: {result.error}"

    assert result.success
    assert result.content
    assert result.model
    assert len(result.content) > 0
    print(f"SUCCESS: Got response: {result.content[:100]}...")


def test_aurora_config_integration():
    config = AuroraConfig.from_env()
    print(f"Config: model_path={config.model_path}, adapter_path={config.adapter_path}")
    print(f"Gen params: max_tokens={config.max_new_tokens}, temp={config.temperature}, top_p={config.top_p}")
    
    err = config.validate()
    if err:
        print(f"SKIP: {err}")
        return
    
    assert config.model_path
    assert config.adapter_path
    assert config.max_new_tokens > 0


if __name__ == "__main__":
    test_aurora_config_integration()
    test_aurora_brain_integration()
    print("Brain integration test completed!")