
import sys
from pathlib import Path

# Fix paths for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import normalize_label

def test_pac_normalization():
    print("--- Testing PAC/APC Normalization ---")
    
    test_cases = [
        ("PAC", "PAC"),
        ("APC", "PAC"),
        ("Atrial Premature Contraction", "PAC"),
        ("atrial premature contraction", "PAC"),
        ("apc", "PAC"),
        ("PAC Bigeminy", "PAC Bigeminy"),
        ("Sinus Rhythm", "Sinus Rhythm")
    ]
    
    for input_val, expected in test_cases:
        result = normalize_label(input_val)
        assert result == expected, f"Failed for '{input_val}': Expected '{expected}', got '{result}'"
        print(f"[PASS] '{input_val}' -> '{result}'")

    print("\nALL PAC NORMALIZATION TESTS PASSED ✅")

if __name__ == "__main__":
    test_pac_normalization()
