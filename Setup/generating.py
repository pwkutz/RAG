import ollama
import sys

class Generator:

    def __init__(self):

        self.LANGUAGE_MODEL: str = r'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
        self.stream: generator = None

    def initialise_chatbot(self, input_query: str, instruction_prompt: str):


        self.stream = ollama.chat(
          model=self.LANGUAGE_MODEL,
          messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
          ],
          stream=True,
        )

    def chatbot_response(self):

        # print the response from the chatbot in real-time
        print('Chatbot response:')
        for chunk in self.stream:
          print(chunk['message']['content'], end='', flush=True)

def main(input_query: str, instruction_prompt: str):

    generator: Generator = Generator()
    generator.initialise_chatbot(input_query = input_query, instruction_prompt = instruction_prompt)
    generator.chatbot_response()