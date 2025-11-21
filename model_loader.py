import os
import joblib
from typing import Tuple, Optional, List

def find_and_load_model(mine_name: str, extra_candidates: Optional[List[str]] = None) -> Tuple[Optional[object], List[str]]:
    """
    Try to find and load a model for `mine_name`.
    Returns (model_or_None, list_of_paths_checked).
    The function checks several candidate paths (ordered).
    """
    checked = []
    candidates = []

    # 1) Common folder layout some apps expect
    candidates.append(os.path.join("models", mine_name, "model.joblib"))
    candidates.append(os.path.join("models", mine_name, "model.pkl"))
    candidates.append(os.path.join("artifacts", mine_name, "model.joblib"))
    candidates.append(os.path.join("artifacts", mine_name, "model.pkl"))

    # 2) Flat names at repo root (matches what you have)
    candidates.append(f"{mine_name}_model.joblib")
    candidates.append(f"{mine_name}_model.pkl")
    candidates.append(f"{mine_name}.joblib")
    candidates.append(f"{mine_name}.pkl")

    # 3) Any extra patterns caller wants to try
    if extra_candidates:
        candidates = extra_candidates + candidates

    # 4) Also search the repo for any file that contains the mine name and looks like a model
    #    (This is a fallback and might be slow if repo is huge)
    try:
        for root, _, files in os.walk("."):
            for f in files:
                lf = f.lower()
                if mine_name.lower() in lf and lf.endswith((".joblib", ".pkl", ".sav")):
                    candidates.append(os.path.join(root, f))
    except Exception:
        # In extremely locked-down runtimes, os.walk might fail; ignore and continue
        pass

    # Deduplicate while preserving order
    seen = set()
    checked_ordered = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            checked_ordered.append(p)

    # Try to load
    for p in checked_ordered:
        checked.append(p)
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                return model, checked
            except Exception:
                # If loading fails, continue to next candidate but record the error in checked
                checked.append(f"[load-failed]{p}")
                continue

    # Nothing loaded
    return None, checked