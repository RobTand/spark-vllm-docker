#!/bin/bash
set -e
# Fix: ignore_keys comes in as a list from config JSON but set operations require a set
# Affects both the |= in __init__ and the -= in _check_received_keys
ROPE_FILE="/usr/local/lib/python3.12/dist-packages/transformers/modeling_rope_utils.py"
sed -i 's/received_keys -= ignore_keys/received_keys -= set(ignore_keys)/' "$ROPE_FILE"
sed -i 's/ignore_keys_at_rope_validation = ignore_keys_at_rope_validation | /ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation) | /' "$ROPE_FILE"
echo "Applied rope validation fix to $ROPE_FILE"
