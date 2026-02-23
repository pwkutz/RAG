# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from setup.dataset import get_dataset as dataset
from setup.embed import main as embedding, show_knowledge
from setup.generating import main as chatbot


def main():

    data: list[str] = dataset()
    input_query: str = 'When does a young cat loose its teeth?'
    retrieved_knowledge: list[tuple[str, float]] = embedding(dataset=data, input_query = input_query) # N most to the query most similar chunks
    instruction_prompt: str = show_knowledge(retrieved_knowledge = retrieved_knowledge) # show the N chunks which are the most similar to the query
    chatbot(input_query= input_query, instruction_prompt= instruction_prompt)




# Press the green button in the gutter to run the script.
if __name__ == '__main__': main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
