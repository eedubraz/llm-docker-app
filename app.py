from fastapi import FastAPI
from transformers import pipeline

## Create a new FastAPI app instance
app=FastAPI()

## Initialize the text generation pipeline
pipe=pipeline("text2text-generation", model="google/flan-t5-small")


@app.get('/')
def home():
    return {"message":"Hello World"}

## Define a function to handle the GET request at '/generate'
@app.get('/generate')
def generate_text(text: str):
    """
    Generates text based on the input text.

    Args:
        text (str): The input text.

    Returns:
        dict: A dictionary containing the generated text.
    """
    # Use the text generation pipeline to generate text based on the input text
    result = pipe(text)
    # Return the generated text as a dictionary
    return {"output": result[0]['generated_text']}

