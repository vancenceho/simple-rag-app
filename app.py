import gradio as gr
import rag_app as rag
import csv
import os
from datetime import datetime

LOG_FILE = "./gradio_cached_examples/15/log.csv"

def log_example(message: str, response: str):
    """
    Log the user input and response to a CSV file -> log.csv

    Parameters
    ------------
    message: str
        User input message
    response: str
        Chatbot response
    
    Returns
    -------
    None

    """
    timestamp = datetime.now().isoformat()
    response = response.replace("\n", "\\n")
    with open(LOG_FILE, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([f'[[\"{message}\", \"{response}\"]]', '', '', timestamp])

def greet(name, intensity):
    """
    Greet the user with a message
    
    Parameters
    ------------
    name: str
        Name of the user
        
    intensity: int
        Intensity of the message

    Returns
    -------
    Greeting message: str

    """
    return "Hello " + name + "!" * intensity

def yes_man(message, history):
    """
    Respond to the user with a "Yes" if the message ends with a question mark

    Parameters
    ------------
    message: str
        User input message

    history: list
        Chat history

    Returns
    -------
    Response message: str
    
    """ 

    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"
    
def raggy_chatbot(message, history):
    """
    Chatbot function that returns a response to the user input message

    Parameters
    ------------
    message: str
        User input message

    history: list
        Chat history

    Returns
    -------
    Response message: str
    
    """

    raggy = rag.rag_app()
    raggy_output = raggy.run(message)
    log_example(message, raggy_output)
    return raggy_output

examples = ["Hello", "When is Python released?", "What are the core concepts of AI?", "What is ML?", "Am I cool?"]


# Gradio Interface
gr.ChatInterface(
    raggy_chatbot,
    chatbot=gr.Chatbot(placeholder="<strong> Raggy Chatbot </strong><br>Ask Me Anything", container=False, scale=7),
    textbox=gr.Textbox(placeholder="Ask me anything about AI, ML or Deep Learning!", container=False, scale=7),
    title="Raggy",
    description="Ask Raggy anything about the world of AI, ML or Deep Learning!",
    theme="soft",
    examples= examples,
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()
