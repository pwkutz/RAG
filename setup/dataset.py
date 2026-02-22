import os.path
import sys
from pathlib import Path

def get_dataset() -> list[str]:

    dataset: list[str] = []
    path_data: Path = Path(os.path.abspath(r'./data/cat-facts.txt'))

    with open(fr'{path_data}', 'r') as file:
        dataset: list[str] = file.readlines()
        return dataset