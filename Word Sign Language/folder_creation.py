
import os
import numpy as np

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Check if DATA_PATH exists, if not, create it
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'yes','no','drink','please','good_luck','help','congratulations','hungry'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 0

for action in actions: 
    action_path = os.path.join(DATA_PATH, action)
    
    # Check if action directory exists, if not, create it
    if not os.path.exists(action_path):
        os.makedirs(action_path)
    
    # Get the maximum directory number inside the action directory
    existing_dirs = np.array(os.listdir(action_path))
    if len(existing_dirs) == 0:
        dirmax = start_folder
    else:
        dirmax = np.max(existing_dirs.astype(int))
    
    # Create new directories
    for sequence in range(0, no_sequences):
        try: 
            new_dir = os.path.join(action_path, str(dirmax + sequence))
            os.makedirs(new_dir)
            print("Created directory:", new_dir)
        except Exception as e:
            print("Error creating directory:", e)
