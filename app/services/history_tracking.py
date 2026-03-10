import json
import os
from datetime import datetime

HISTORY_FILE = "data/dataset_history.json"


def load_dataset_history():
    """Load dataset history from file."""

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_dataset_summary(summary):
    """Save a dataset summary to history."""

    # Create data folder if missing
    if not os.path.exists("data"):
        os.makedirs("data")

    history = load_dataset_history()

    # Add timestamp automatically
    summary["timestamp"] = datetime.now().isoformat()

    history.append(summary)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)