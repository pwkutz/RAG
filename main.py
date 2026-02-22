# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from setup.dataset import get_dataset as dataset
from setup.embed import main as embedding

def main():

    data: list[str] = dataset()
    embedding(dataset=data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__': main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
