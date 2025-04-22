# kimchi.py
# For converting Python 2 pickles to Python 3

import os
import dill
import pickle


def convert(old_pkl: str) -> str:
    """
    Convert a Python 2 pickle to Python 3

    old_pkl = path to Python 2 pickle file
    """
    # Make a name for the new pickle
    new_pkl = os.path.splitext(old_pkl)[0] + "_p3.pkl"

    print(new_pkl)

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)
