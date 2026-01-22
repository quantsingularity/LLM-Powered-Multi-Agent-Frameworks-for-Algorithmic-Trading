import os
import sys

# Ensure repo's code/ is importable as top-level modules (data, agents, etc.)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE = os.path.join(ROOT, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)
