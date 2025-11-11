#!/usr/bin/env python3
"""
Temporary script to inspect existing vector DB metadata
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load metadata
metadata_path = "output/vector/embeddings_v1_metadata.pkl"

with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

print("=== Vector DB Metadata ===")
print(f"\nDimension: {metadata.get('dimension')}")
print(f"Num Notices: {len(metadata.get('notice_ids', []))}")
print(f"Num Companies: {len(metadata.get('company_ids', []))}")

# Check ID format
notice_ids = metadata.get('notice_ids', [])
company_ids = metadata.get('company_ids', [])

print(f"\n=== Notice ID Samples (first 5) ===")
for i, nid in enumerate(notice_ids[:5]):
    print(f"{i}: {nid} (type: {type(nid).__name__})")

print(f"\n=== Company ID Samples (first 5) ===")
for i, cid in enumerate(company_ids[:5]):
    print(f"{i}: {cid} (type: {type(cid).__name__})")

print(f"\n=== ID to Index Mapping (first 3) ===")
notice_id_to_idx = metadata.get('notice_id_to_idx', {})
for i, (k, v) in enumerate(list(notice_id_to_idx.items())[:3]):
    print(f"{k} -> {v}")
