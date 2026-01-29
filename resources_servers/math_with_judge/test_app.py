#!/usr/bin/env python3
"""Test script to verify the modified app.py works with original data format."""

import json
import sys
from pydantic import BaseModel
from typing import Optional

# Test backward compatibility with existing data format (without new fields)
def test_backward_compatibility():
    """Test that requests without llama8b_solve_rate, domain, source work."""
    print("Testing backward compatibility...")
    
    # Import the models
    sys.path.insert(0, '/Users/user/Documents/hf-AI/RL/3rdparty/Gym-workspace/gym')
    from resources_servers.math_with_judge.app import (
        LibraryJudgeMathRunRequest,
        LibraryJudgeMathVerifyRequest,
        LibraryJudgeMathResourcesServerConfig,
        LibraryJudgeMathResourcesServer,
        ALLOWED_TAGS
    )
    
    # Test 1: Create request without optional fields (like original format)
    try:
        request_data = {
            "question": "What is 2+2?",
            "expected_answer": "4"
        }
        request = LibraryJudgeMathRunRequest(**request_data)
        assert request.question == "What is 2+2?"
        assert request.expected_answer == "4"
        assert request.llama8b_solve_rate == 0.5  # Default
        assert request.domain == "unknown"  # Default
        assert request.source == "unknown"  # Default
        print("✓ Backward compatibility test 1: PASSED (original format works)")
    except Exception as e:
        print(f"✗ Backward compatibility test 1: FAILED - {e}")
        return False
    
    # Test 2: Create request with new fields
    try:
        request_data_new = {
            "question": "What is 2+2?",
            "expected_answer": "4",
            "llama8b_solve_rate": 0.3,
            "domain": "arithmetic",
            "source": "synthetic"
        }
        request_new = LibraryJudgeMathRunRequest(**request_data_new)
        assert request_new.llama8b_solve_rate == 0.3
        assert request_new.domain == "arithmetic"
        assert request_new.source == "synthetic"
        print("✓ Backward compatibility test 2: PASSED (new fields work)")
    except Exception as e:
        print(f"✗ Backward compatibility test 2: FAILED - {e}")
        return False
    
    # Test 3: Verify allowed tags are properly defined
    try:
        assert "axiom" in ALLOWED_TAGS
        assert "def" in ALLOWED_TAGS
        assert "theorem" in ALLOWED_TAGS
        print(f"✓ Allowed tags test: PASSED ({len(ALLOWED_TAGS)} tags defined)")
    except Exception as e:
        print(f"✗ Allowed tags test: FAILED - {e}")
        return False
    
    return True


def test_reward_computations():
    """Test the reward computation methods."""
    print("\nTesting reward computation methods...")
    
    sys.path.insert(0, '/Users/user/Documents/hf-AI/RL/3rdparty/Gym-workspace/gym')
    from resources_servers.math_with_judge.app import LibraryJudgeMathResourcesServer
    
    # Create a mock server instance - we can't fully initialize without config
    # but we can test the computation methods are accessible
    try:
        # Test that methods exist and are callable
        assert hasattr(LibraryJudgeMathResourcesServer, '_compute_format_reward')
        assert hasattr(LibraryJudgeMathResourcesServer, '_compute_axiom_reward')
        assert hasattr(LibraryJudgeMathResourcesServer, '_compute_difficulty_multiplier')
        assert hasattr(LibraryJudgeMathResourcesServer, '_log_trace')
        print("✓ All new reward methods are defined")
    except Exception as e:
        print(f"✗ Reward computation methods test: FAILED - {e}")
        return False
    
    return True


def test_config_defaults():
    """Test that config has proper defaults."""
    print("\nTesting configuration defaults...")
    
    sys.path.insert(0, '/Users/user/Documents/hf-AI/RL/3rdparty/Gym-workspace/gym')
    from resources_servers.math_with_judge.app import LibraryJudgeMathResourcesServerConfig
    from nemo_gym.config_types import ModelServerRef
    
    try:
        # Test default weight values
        config_dict = {
            "judge_model_server": {"name": "test", "type": "responses_api_models"},
            "judge_responses_create_params": {"input": []}
        }
        
        # We can't fully construct the config without proper types, but we can verify
        # that the fields are defined with defaults in the class
        fields = LibraryJudgeMathResourcesServerConfig.model_fields
        
        assert 'format_weight' in fields
        assert 'outcome_weight' in fields
        assert 'axiom_weight' in fields
        assert 'coherence_weight' in fields
        assert 'relevance_weight' in fields
        assert 'efficiency_weight' in fields
        assert 'use_difficulty_scaling' in fields
        assert 'difficulty_scaling_cap' in fields
        assert 'save_traces' in fields
        assert 'trace_dir' in fields
        
        print("✓ All configuration fields are defined")
        
        # Check defaults
        assert fields['format_weight'].default == 0.1
        assert fields['outcome_weight'].default == 2.0
        assert fields['axiom_weight'].default == 0.1
        assert fields['coherence_weight'].default == 0.2
        assert fields['relevance_weight'].default == 0.1
        assert fields['efficiency_weight'].default == 0.05
        assert fields['use_difficulty_scaling'].default == True
        assert fields['difficulty_scaling_cap'].default == 4.0
        assert fields['save_traces'].default == True
        assert fields['trace_dir'].default == "logs/training_traces"
        
        print("✓ All configuration defaults are correct")
    except Exception as e:
        print(f"✗ Configuration defaults test: FAILED - {e}")
        return False
    
    return True


def test_response_model():
    """Test that response model includes new fields."""
    print("\nTesting response model...")
    
    sys.path.insert(0, '/Users/user/Documents/hf-AI/RL/3rdparty/Gym-workspace/gym')
    from resources_servers.math_with_judge.app import LibraryJudgeMathVerifyResponse
    
    try:
        fields = LibraryJudgeMathVerifyResponse.model_fields
        
        # Check new response fields exist
        assert 'format_reward' in fields
        assert 'outcome_reward' in fields
        assert 'axiom_reward' in fields
        assert 'coherence_reward' in fields
        assert 'relevance_reward' in fields
        assert 'efficiency_reward' in fields
        assert 'difficulty_multiplier' in fields
        assert 'scaled_outcome_reward' in fields
        
        print("✓ All new response fields are defined")
    except Exception as e:
        print(f"✗ Response model test: FAILED - {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Running verification tests for modified app.py")
    print("=" * 60)
    
    tests = [
        test_backward_compatibility,
        test_reward_computations,
        test_config_defaults,
        test_response_model,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED - Code is backward compatible!")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} test(s) failed")
        sys.exit(1)
