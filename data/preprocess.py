################################################################################
# Driver Script to preprocess the raw pdf documents into mschine readable text #
################################################################################

import os


local_path = os.path.join(os.path.dirname(__file__), "clean")
os.makedirs(local_path, exist_ok=True)