"""A utility function to build a tree representation of a directory structure."""
from pathlib import Path


def build_tree(path: Path):
    """
    Recursively build a tree of the folder as a dict.
    Files are listed with None values, directories as nested dicts.
    """
    tree = {}
    try:
        for p in sorted(path.iterdir()):
            if p.is_dir():
                tree[p.name] = build_tree(p)
            else:
                tree[p.name] = None
        return tree
    except PermissionError:
        return {"error": "Permission denied"}
