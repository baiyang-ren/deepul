import numpy as np
import os
import sys

def setup_notebook_environment():
    """
    Setup function for Jupyter notebook environment.
    This replaces the problematic __file__ usage with notebook-compatible code.
    """
    # Get current working directory
    notebook_dir = os.getcwd()
    
    # Try to find and navigate to the hw1 directory
    if 'deepul' in notebook_dir:
        # If we're already in the deepul directory, go to the hw1 directory
        if 'hw1' not in notebook_dir:
            hw1_path = os.path.join(notebook_dir, 'homeworks', 'hw1')
            if os.path.exists(hw1_path):
                os.chdir(hw1_path)
    else:
        # Try to find the deepul directory by walking up the directory tree
        current_dir = notebook_dir
        while current_dir != os.path.dirname(current_dir):
            deepul_path = os.path.join(current_dir, 'deepul')
            if os.path.exists(deepul_path):
                hw1_path = os.path.join(deepul_path, 'homeworks', 'hw1')
                if os.path.exists(hw1_path):
                    os.chdir(hw1_path)
                    break
            current_dir = os.path.dirname(current_dir)
    
    # Add the deepul package to the path
    deepul_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    if deepul_path not in sys.path:
        sys.path.append(deepul_path)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Added to sys.path: {deepul_path}")

# Run the setup
setup_notebook_environment()

# Now import the required modules
try:
    from deepul.hw1_helper import (
        # Q1
        visualize_q1_data,
        q1_sample_data_1,
        q1_sample_data_2,
        q1_save_results,
        # Q2
        q2a_save_results,
        q2b_save_results,
        visualize_q2a_data,
        visualize_q2b_data,
        # Q3
        q3ab_save_results,
        q3c_save_results,
        # Q4
        q4a_save_results,
        q4b_save_results,
        # Q5
        visualize_q5_data,
        q5a_save_results,
        # Q6
        visualize_q6_data,
        q6a_save_results,
    )
    print("Successfully imported all required modules!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure the deepul package is properly installed.") 