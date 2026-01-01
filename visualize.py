#!/usr/bin/env python3
"""
Wrapper script to run visualize_results from any directory
"""
import sys
import os

# Change to project root if needed
if os.path.exists('federated_learning.py'):
    # Already in project root
    pass
elif os.path.exists('../federated_learning.py'):
    os.chdir('..')
elif os.path.exists('../../federated_learning.py'):
    os.chdir('../..')

# Now import and run
from tools.visualize_results import main

if __name__ == '__main__':
    main()
