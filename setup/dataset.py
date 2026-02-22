import os.path
import sys
from pathlib import Path

def get_dataset():

    dataset = []
    path_data: Path = Path(os.path.abspath(r'./data/cat-facts.txt'))
    with open(fr'{path_data}', 'r') as file:
        dataset = file.readlines()
        print(f'Loaded {len(dataset)} entries')