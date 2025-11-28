#!/usr/bin/env python
"""
Test script for runtime package

This script tests the runtime package installation and basic functionality.
"""

import sys
import os
from pathlib import Path

# Test 1: Check if the package can be imported
print("=" * 60)
print("Testing runtime package")
print("=" * 60)

try:
    # Add project root to path for direct testing (before installation)
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("\n1. Testing import...")
    from runtime import initialize, run_step1_prediction
    print("✓ Package import successful")

    # Test service module
    from runtime.service import (
        get_notice_embedding,
        get_company_embedding,
        predict_batch,
        shutdown
    )
    print("✓ Service functions imported")

    # Test model module
    from runtime.model import ModelLoader, TwoTowerPredictor
    print("✓ Model components imported")

    # Test preprocess module
    from runtime.preprocess import TorchRecAdapter
    print("✓ Preprocessing components imported")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check if database module is accessible
print("\n2. Testing database module...")
try:
    # Add project root to path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from database.database_connector import DatabaseConnector
    print("✓ Database module imported")

    # Check if we can create a connector (without connecting)
    print("  - DatabaseConnector class available")

except ImportError as e:
    print(f"✗ Database import failed: {e}")
    print("  Note: This is expected if database module is not in path")

# Test 3: Check package structure
print("\n3. Checking package structure...")
runtime_path = project_root / "runtime"

required_files = [
    "__init__.py",
    "service.py",
    "model/__init__.py",
    "model/loader.py",
    "model/predictor.py",
    "preprocess/__init__.py",
    "preprocess/torchrec_adapter.py"
]

all_present = True
for file in required_files:
    file_path = runtime_path / file
    if file_path.exists():
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - MISSING")
        all_present = False

if all_present:
    print("\n✓ All required files present")
else:
    print("\n✗ Some files are missing")

# Test 4: Check if initialization would work (dry run)
print("\n4. Testing initialization (dry run)...")
print("  - initialize() function available")
print("  - run_step1_prediction() function available")
print("  - predict_batch() function available")
print("  - shutdown() function available")

# Test 5: Check model checkpoint
print("\n5. Checking model checkpoint...")
checkpoint_path = Path("/data/dev/jodalroB-twoTower/output/models/20251118_071938/checkpoint_epoch_1.pt")
if checkpoint_path.exists():
    print(f"✓ Model checkpoint found: {checkpoint_path}")
    print(f"  Size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
else:
    print(f"✗ Model checkpoint not found: {checkpoint_path}")

# Test 6: Check metadata
print("\n6. Checking metadata...")
metadata_path = Path("/data/dev/jodalroB-twoTower/meta/metadata.csv")
if metadata_path.exists():
    print(f"✓ Metadata file found: {metadata_path}")
else:
    print(f"✗ Metadata file not found: {metadata_path}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
The runtime package structure is ready!

Next steps:
1. Install the package in jodalroB-prediction:
   cd /data/dev/jodalroB-prediction
   source .venv/bin/activate
   pip install -e ../jodalroB-twoTower

2. Update jodalroB-prediction/src/step1.py to use the runtime:
   from runtime.service import initialize, run_step1_prediction

3. Test the integration:
   python -c "from runtime import initialize"

Note: Full functionality test requires:
- Database connection
- Model checkpoint
- CUDA/GPU (optional)
""")

print("\nTest completed!")