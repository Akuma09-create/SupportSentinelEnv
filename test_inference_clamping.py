"""
Test script to validate defensive score clamping in inference.py.
Validates all edge cases without requiring the environment server running.
"""
import sys
from pathlib import Path

# Import the functions we're testing
from inference import (
    clamp_score,
    parse_reward_score,
    fmt_reward,
    fmt_bool,
    SCORE_MIN,
    SCORE_MAX,
)

def test_clamp_score():
    """Test clamp_score() with various edge cases."""
    print("\n" + "="*60)
    print(" TEST 1: clamp_score() Edge Cases")
    print("="*60)
    
    test_cases = [
        # (input, expected_output, description)
        (0.0, SCORE_MIN, "Zero → min boundary"),
        (1.0, SCORE_MAX, "One → max boundary"),
        (-0.5, SCORE_MIN, "Negative → min"),
        (1.5, SCORE_MAX, "Greater than 1 → max"),
        (100.0, SCORE_MAX, "Large value → max"),
        (-100.0, SCORE_MIN, "Large negative → min"),
        (0.5, 0.5, "Valid middle value"),
        (SCORE_MIN, SCORE_MIN, "Exact minimum"),
        (SCORE_MAX, SCORE_MAX, "Exact maximum"),
        ("invalid", SCORE_MIN, "Invalid string → min"),
        (None, SCORE_MIN, "None → min"),
        ([], SCORE_MIN, "Invalid list → min"),
    ]
    
    all_passed = True
    for input_val, expected, description in test_cases:
        try:
            result = clamp_score(input_val)
            passed = abs(result - expected) < 0.001
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {description:30} | Input: {str(input_val):15} → {result:.2f}")
            if not passed:
                print(f"       Expected {expected:.2f}, got {result:.2f}")
                all_passed = False
        except Exception as e:
            print(f"❌ FAIL | {description:30} | Exception: {e}")
            all_passed = False
    
    return all_passed


def test_parse_reward_score():
    """Test parse_reward_score() with various reward object types."""
    print("\n" + "="*60)
    print(" TEST 2: parse_reward_score() Extraction & Clamping")
    print("="*60)
    
    test_cases = [
        # (input_reward, expected_score, expected_error, description)
        (
            {"score": 0.5, "last_action_error": None},
            0.5,
            None,
            "Valid dict with score=0.5"
        ),
        (
            {"score": 1.5, "last_action_error": "Some error"},
            SCORE_MAX,
            "Some error",
            "Dict with score > 1 → clamped to max"
        ),
        (
            {"score": -0.5},
            SCORE_MIN,
            None,
            "Dict with negative score → clamped to min"
        ),
        (
            {"score": 0.0},
            SCORE_MIN,
            None,
            "Dict with score=0 → clamped to min"
        ),
        (
            0.5,
            0.5,
            None,
            "Float input (not dict)"
        ),
        (
            1.0,
            SCORE_MAX,
            None,
            "Float input=1.0 → clamped to max"
        ),
        (
            -0.1,
            SCORE_MIN,
            None,
            "Float input negative → clamped to min"
        ),
        (
            {},
            SCORE_MIN,
            None,
            "Empty dict → defaults to min"
        ),
        (
            None,
            SCORE_MIN,
            None,
            "None input → defaults to min"
        ),
    ]
    
    all_passed = True
    for reward_obj, expected_score, expected_error, description in test_cases:
        try:
            score, error = parse_reward_score(reward_obj)
            score_match = abs(score - expected_score) < 0.001
            error_match = error == expected_error
            passed = score_match and error_match
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {description:40} | Score: {score:.2f} | Error: {error}")
            if not passed:
                if not score_match:
                    print(f"       Expected score {expected_score:.2f}, got {score:.2f}")
                if not error_match:
                    print(f"       Expected error '{expected_error}', got '{error}'")
                all_passed = False
        except Exception as e:
            print(f"❌ FAIL | {description:40} | Exception: {e}")
            all_passed = False
    
    return all_passed


def test_fmt_reward():
    """Test fmt_reward() formatting with clamping."""
    print("\n" + "="*60)
    print(" TEST 3: fmt_reward() Formatting with Clamping")
    print("="*60)
    
    test_cases = [
        # (input, description)
        (0.0, "Zero → min"),
        (1.0, "One → max"),
        (0.5, "Middle value"),
        (1.5, "Greater than 1"),
        (-0.5, "Negative"),
        (SCORE_MAX, "Exact max"),
        (SCORE_MIN, "Exact min"),
    ]
    
    all_passed = True
    for value, description in test_cases:
        try:
            result = fmt_reward(value)
            # Parse the result to verify it's a valid number string
            parsed = float(result)
            valid_range = SCORE_MIN <= parsed <= SCORE_MAX
            passed = valid_range and len(result.split('.')[1]) == 2  # Check 2 decimal places
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {description:20} | Input: {value:6.2f} → Formatted: {result}")
            if not passed:
                print(f"       Must be in range [{SCORE_MIN}, {SCORE_MAX}] with 2 decimals, got {result}")
                all_passed = False
        except Exception as e:
            print(f"❌ FAIL | {description:20} | Exception: {e}")
            all_passed = False
    
    return all_passed


def test_fmt_bool():
    """Test fmt_bool() formatting."""
    print("\n" + "="*60)
    print(" TEST 4: fmt_bool() Formatting")
    print("="*60)
    
    test_cases = [
        (True, "true", "True → 'true'"),
        (False, "false", "False → 'false'"),
    ]
    
    all_passed = True
    for value, expected, description in test_cases:
        try:
            result = fmt_bool(value)
            passed = result == expected
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {description:20} | Result: '{result}'")
            if not passed:
                print(f"       Expected '{expected}', got '{result}'")
                all_passed = False
        except Exception as e:
            print(f"❌ FAIL | {description:20} | Exception: {e}")
            all_passed = False
    
    return all_passed


def test_integration():
    """Integration test: simulate a full reward pipeline."""
    print("\n" + "="*60)
    print(" TEST 5: Integration - Full Reward Pipeline")
    print("="*60)
    
    # Simulate various reward scenarios
    test_scenarios = [
        {
            "name": "Scenario A: Perfect SLA (1.0)",
            "reward_obj": {"score": 1.0, "last_action_error": None},
            "expected_final": "0.99",
        },
        {
            "name": "Scenario B: Failed validation (0.0)",
            "reward_obj": {"score": 0.0, "last_action_error": "Validation failed"},
            "expected_final": "0.01",
        },
        {
            "name": "Scenario C: Normal success (0.75)",
            "reward_obj": {"score": 0.75, "last_action_error": None},
            "expected_final": "0.75",
        },
        {
            "name": "Scenario D: Extreme value (2.5)",
            "reward_obj": {"score": 2.5},
            "expected_final": "0.99",
        },
    ]
    
    all_passed = True
    for scenario in test_scenarios:
        try:
            # Extract score
            score, error = parse_reward_score(scenario["reward_obj"])
            
            # Format for logging
            formatted = fmt_reward(score)
            
            # Check result
            passed = formatted == scenario["expected_final"]
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {scenario['name']:40} | Final: {formatted}")
            
            if not passed:
                print(f"       Expected {scenario['expected_final']}, got {formatted}")
                all_passed = False
        except Exception as e:
            print(f"❌ FAIL | {scenario['name']:40} | Exception: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  INFERENCE.PY DEFENSIVE CLAMPING TEST SUITE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    print(f"\nScore Safety Bounds: [{SCORE_MIN}, {SCORE_MAX}]")
    print("All values must be strictly in (0, 1) for validation compliance.")
    
    results = {
        "clamp_score()": test_clamp_score(),
        "parse_reward_score()": test_parse_reward_score(),
        "fmt_reward()": test_fmt_reward(),
        "fmt_bool()": test_fmt_bool(),
        "Integration": test_integration(),
    }
    
    # Summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Inference clamping is bulletproof!")
        print("="*60)
        return 0
    else:
        print("❌ SOME TESTS FAILED - Review output above")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
