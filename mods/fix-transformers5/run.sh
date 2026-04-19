#!/bin/bash
set -e
pip install --no-deps 'transformers>=5.0.0,<5.1.0' 2>&1 | tail -1
echo "Upgraded transformers to 5.x"
