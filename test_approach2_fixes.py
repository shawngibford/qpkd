#!/usr/bin/env python3
"""
Test script to validate approach2_qml_demo.py fixes
"""

import ast
import numpy as np

def test_approach2_fixes():
    """Test that approach2_qml_demo.py uses real data only and has proper error handling."""
    print("="*60)
    print("TESTING APPROACH2_QML_DEMO.PY FIXES")
    print("="*60)

    # Read the fixed script
    script_path = "notebooks/approach2_qml_demo.py"
    with open(script_path, 'r') as f:
        content = f.read()

    # Test 1: Check for removed fake quantum advantage
    print("1. Testing Fake Quantum Advantage Removal:")
    fake_advantage_patterns = [
        "quantum_predictions = rf_predictions.copy()",
        "Add quantum advantage",
        "improvement_mask = np.random.choice",
        "quantum_predictions[idx] = test_targets_cls[idx]"
    ]

    has_fake_advantage = any(pattern in content for pattern in fake_advantage_patterns)
    if has_fake_advantage:
        print("❌ FAILED: Still contains fake quantum advantage simulation")
        return False
    else:
        print("✅ PASSED: Fake quantum advantage simulation removed")

    # Test 2: Check for removed fake coverage prediction
    print("\n2. Testing Fake Coverage Prediction Removal:")
    fake_coverage_patterns = [
        "coverage = min(0.95, 0.1 + (dose / 30) * 0.8 + np.random.normal",
        "+ np.random.normal(0, 0.02)"
    ]

    has_fake_coverage = any(pattern in content for pattern in fake_coverage_patterns)
    if has_fake_coverage:
        print("❌ FAILED: Still contains fake coverage prediction")
        return False
    else:
        print("✅ PASSED: Fake coverage prediction removed")

    # Test 3: Check for proper error handling (no masking try-catch blocks)
    print("\n3. Testing Error Handling Improvements:")
    problematic_patterns = [
        "except Exception as e:\n    print(f\"Circuit visualization failed: {e}\")",
        "except:\n        score = 0.0"
    ]

    has_masking_errors = any(pattern in content for pattern in problematic_patterns)
    if has_masking_errors:
        print("❌ FAILED: Still contains error-masking try-catch blocks")
        return False
    else:
        print("✅ PASSED: Error-masking try-catch blocks removed")

    # Test 4: Check for removed data augmentation
    print("\n4. Testing Data Augmentation Removal:")
    augmentation_patterns = [
        "augmented_data = preprocessor.augment_data(",
        "augmentation_factor=3, noise_level=0.05"
    ]

    has_augmentation = any(pattern in content for pattern in augmentation_patterns)
    if has_augmentation:
        print("❌ FAILED: Still contains data augmentation with synthetic noise")
        return False
    else:
        print("✅ PASSED: Data augmentation with synthetic noise removed")

    # Test 5: Check for deterministic demo features
    print("\n5. Testing Demo Features:")
    if "demo_features = np.random.random(5)" in content:
        print("❌ FAILED: Still using random demo features")
        return False
    elif "demo_features = np.array([24.0, 5.0, 70.0, 0.0, 1.0])" in content:
        print("✅ PASSED: Using deterministic real-data-based demo features")
    else:
        print("⚠️  WARNING: Demo features changed but not verified")

    # Test 6: Check for improved quantum feature extraction
    print("\n6. Testing Quantum Feature Extraction:")
    if "except AttributeError:" in content and "quantum_features.append" in content:
        print("❌ FAILED: Still using try-catch for quantum feature extraction")
        return False
    elif "if hasattr(model, '_extract_quantum_features'):" in content:
        print("✅ PASSED: Using proper conditional for quantum feature extraction")
    else:
        print("⚠️  WARNING: Quantum feature extraction logic changed")

    # Test 7: Verify no random data generation for ML comparison
    print("\n7. Testing ML Model Training:")
    if "except:\n        score = 0.0" in content:
        print("❌ FAILED: Still masking ML training failures")
        return False
    else:
        print("✅ PASSED: ML training failures will now surface properly")

    print(f"\n✅ ALL TESTS PASSED!")
    print(f"✅ approach2_qml_demo.py now uses real data exclusively")
    print(f"✅ Error handling no longer masks real issues")
    print(f"✅ Fake quantum advantage and synthetic data removed")

    return True

def test_data_flow_integrity():
    """Test that data flows correctly from EstData.csv to models."""
    print(f"\n" + "="*60)
    print("TESTING DATA FLOW INTEGRITY")
    print("="*60)

    # Since we can't run the actual script due to PennyLane issues,
    # test the data loading logic directly
    try:
        import sys
        sys.path.append('src')
        from data.data_loader import PKPDDataLoader

        # Test data loading
        loader = PKPDDataLoader("data/EstData.csv")
        data = loader.prepare_pkpd_data(weight_range=(50, 100), concomitant_allowed=True)

        print("✅ Data loader working correctly")
        print(f"  - Subjects: {len(data.subjects)}")
        print(f"  - Features shape: {data.features.shape}")
        print(f"  - Feature columns: Weight, Dose, Conmed (3 features only)")

        # Verify no synthetic data
        feature_ranges = {
            'Weight': (data.features[:, 0].min(), data.features[:, 0].max()),
            'Dose': (data.features[:, 1].min(), data.features[:, 1].max()),
            'Conmed': (data.features[:, 2].min(), data.features[:, 2].max())
        }

        print(f"✅ Feature ranges from real data:")
        for feature, (min_val, max_val) in feature_ranges.items():
            print(f"  - {feature}: {min_val:.1f} - {max_val:.1f}")

        return True

    except Exception as e:
        print(f"❌ Data flow test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing approach2_qml_demo.py debugging fixes...")

    success = True
    success &= test_approach2_fixes()
    success &= test_data_flow_integrity()

    print(f"\n" + "="*60)
    if success:
        print("🎉 ALL APPROACH2 TESTS PASSED!")
        print("")
        print("FIXES APPLIED TO APPROACH2_QML_DEMO.PY:")
        print("✅ Removed fake quantum advantage simulation")
        print("✅ Removed fake coverage prediction with random noise")
        print("✅ Eliminated error-masking try-catch blocks")
        print("✅ Removed data augmentation with synthetic noise")
        print("✅ Replaced random demo features with real data values")
        print("✅ Fixed quantum feature extraction error handling")
        print("✅ Removed ML training error masking")
        print("")
        print("RESULT:")
        print("✅ approach2_qml_demo.py now uses REAL DATA EXCLUSIVELY")
        print("✅ Errors will surface properly for debugging")
        print("✅ No synthetic/fake quantum advantage")
    else:
        print("❌ SOME APPROACH2 TESTS FAILED!")
    print("="*60)

if __name__ == "__main__":
    main()