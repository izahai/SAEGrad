import sys
import os
import logging

# Ensure we can import from the mend directory by adding the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mesd.util import collect_linear_layers, print_linear_report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    layers = collect_linear_layers()
    print_linear_report(layers)