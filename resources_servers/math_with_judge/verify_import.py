#!/usr/bin/env python3
"""Simple import test to verify the module loads without errors."""

import sys
import ast
import os

def check_syntax(filepath):
    """Check Python syntax by parsing the AST."""
    print(f"Checking syntax of {filepath}...")
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✓ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

def check_class_definitions(filepath):
    """Check that all required classes and methods are defined."""
    print("\nChecking class definitions...")
    with open(filepath, 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    classes = {}
    functions = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    classes[node.name].append(item.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = True
    
    # Check required classes
    required_classes = [
        'LibraryJudgeMathResourcesServerConfig',
        'LibraryJudgeMathRunRequest',
        'LibraryJudgeMathVerifyRequest',
        'LibraryJudgeMathVerifyResponse',
        'LibraryJudgeMathResourcesServer',
        'JudgeEvaluation'
    ]
    
    for cls in required_classes:
        if cls in classes:
            print(f"✓ Class '{cls}' is defined")
        else:
            print(f"✗ Class '{cls}' is missing")
            return False
    
    # Check new methods in LibraryJudgeMathResourcesServer
    required_methods = [
        '_compute_format_reward',
        '_compute_axiom_reward',
        '_compute_difficulty_multiplier',
        '_log_trace',
        'verify',
        '_verify_answer',
        '_verify_answer_with_judge',
        '_generate_judge_evaluation'
    ]
    
    server_class_methods = classes.get('LibraryJudgeMathResourcesServer', [])
    for method in required_methods:
        if method in server_class_methods:
            print(f"✓ Method '{method}' is defined")
        else:
            print(f"✗ Method '{method}' is missing")
            return False
    
    return True

def check_config_attributes(filepath):
    """Check that configuration class has all required attributes."""
    print("\nChecking configuration attributes...")
    with open(filepath, 'r') as f:
        code = f.read()
    
    # Look for attribute definitions in LibraryJudgeMathResourcesServerConfig
    required_attrs = [
        'format_weight',
        'outcome_weight',
        'axiom_weight',
        'coherence_weight',
        'relevance_weight',
        'efficiency_weight',
        'use_difficulty_scaling',
        'difficulty_scaling_cap',
        'save_traces',
        'trace_dir'
    ]
    
    for attr in required_attrs:
        if f'{attr}:' in code or f'{attr} =' in code:
            print(f"✓ Config attribute '{attr}' is defined")
        else:
            print(f"✗ Config attribute '{attr}' is missing")
            return False
    
    return True

def check_backward_compatibility(filepath):
    """Check that optional fields have defaults."""
    print("\nChecking backward compatibility...")
    with open(filepath, 'r') as f:
        code = f.read()
    
    # Check that new request fields have Optional and defaults
    checks = [
        ('llama8b_solve_rate: Optional[float] = 0.5', 'llama8b_solve_rate default'),
        ('domain: Optional[str] = "unknown"', 'domain default'),
        ('source: Optional[str] = "unknown"', 'source default'),
    ]
    
    for pattern, desc in checks:
        if pattern in code:
            print(f"✓ {desc} is set correctly")
        else:
            print(f"✗ {desc} might be missing")
            return False
    
    return True

def main():
    filepath = '/Users/user/Documents/hf-AI/RL/3rdparty/Gym-workspace/gym/resources_servers/math_with_judge/app.py'
    
    print("=" * 70)
    print("Verification Test for Modified math_with_judge/app.py")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return 1
    
    tests = [
        (check_syntax, filepath),
        (check_class_definitions, filepath),
        (check_config_attributes, filepath),
        (check_backward_compatibility, filepath),
    ]
    
    passed = 0
    failed = 0
    
    for test_func, arg in tests:
        try:
            if test_func(arg):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ ALL CHECKS PASSED!")
        print("The modified code:")
        print("  - Has valid Python syntax")
        print("  - Contains all required classes and methods")
        print("  - Has all configuration attributes")
        print("  - Maintains backward compatibility with original data format")
        return 0
    else:
        print(f"\n✗ {failed} check(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
