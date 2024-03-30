import os
import sys

# Construct the path to the parent directory of the parent directory of imagenet_ddp.py
parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

# Add the constructed path to sys.path
sys.path.append(parent_parent_dir)

# Print the updated sys.path to verify the addition
print(sys.path)

from blox_enumerator import BloxEnumerate

