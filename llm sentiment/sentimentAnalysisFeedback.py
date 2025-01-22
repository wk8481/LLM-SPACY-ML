import os
import sys
import torch
import warnings
import pandas as pd
from groq import Groq
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize the Groq client
groq_client = Groq(api_key="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")

# Available Hugging Face models
llm_options = {
    "groq": "groq",
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "gpt-neo": "EleutherAI/gpt-neo-2.7B",
    "minilm": "deepset/minilm-uncased-squad2",
    "bert": "csarron/bert-base-uncased-squad-v1"
}

# Function to load the NER and Sentiment analysis models
def load_aspect_model():
    # Load an NER model for aspect extraction
    aspect_extractor = pipeline('ner', model="dbmdz/bert-large-cased-finetuned-conll03-english")

    # Load a sentiment analysis model
    sentiment_analyzer = pipeline("sentiment-analysis")

    return aspect_extractor, sentiment_analyzer

# Function to extract aspects and sentiments
def extract_aspects_and_sentiments(review_text, aspect_extractor, sentiment_analyzer):
    # Extract aspects (NER)
    entities = aspect_extractor(review_text)
    aspects = {entity['word']: entity['entity'] for entity in entities}

    # Analyze sentiment for the entire review
    sentiment = sentiment_analyzer(review_text)

    return aspects, sentiment

# Function to generate a custom response based on aspects and sentiment
def generate_response(aspects, sentiment):
    if sentiment == "POSITIVE":
        response = "Thank you for your positive feedback! We're glad you appreciated the following aspects: "
        response += ", ".join(aspects.keys()) if aspects else "various parts of our service."
        response += ". We look forward to serving you again!"
    elif sentiment == "NEGATIVE":
        response = "We apologize for the inconvenience caused. We take the following issues seriously: "
        response += ", ".join(aspects.keys()) if aspects else "several parts of our service."
        response += ". We'll work hard to address your concerns."
    else:
        response = "Thank you for your feedback!"

    return response

# Updated Hugging Face response function
def get_huggingface_aspect_sentiment(review_text, aspect_extractor, sentiment_analyzer):
    try:
        # Extract aspects and sentiment
        aspects, sentiment = extract_aspects_and_sentiments(review_text, aspect_extractor, sentiment_analyzer)

        # Format response
        sentiment_label = sentiment[0]['label']
        response = {
            "Aspects": aspects,
            "Sentiment": sentiment_label,
            "Confidence": sentiment[0]['score'],
            "GeneratedResponse": generate_response(aspects, sentiment_label)
        }
        return response
    except Exception as e:
        print(f"Error with Hugging Face model: {e}")
        return {"Error": str(e)}

# Updated Groq response function
def get_groq_aspect_sentiment(review_text, chat_history):
    # Modify the prompt to explicitly ask for aspects and sentiments
    messages = chat_history + [
        {"role": "user", "content": f"Please extract the aspects and sentiments from the following review: '{review_text}'"}
    ]

    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )

    # Parse the response
    response_content = chat_completion.choices[0].message.content
    # Assuming Groq returns aspects and sentiments as text
    aspects = {"General": "Sentiment-related aspect"}  # Replace with actual parsing logic for aspects
    sentiment = "POSITIVE" if "positive" in response_content.lower() else "NEGATIVE"  # Basic sentiment parsing

    # Generate a response
    generated_response = generate_response(aspects, sentiment)

    return {"Aspects": aspects, "Sentiment": sentiment, "GeneratedResponse": generated_response}

# Updated review processing function
def process_reviews_from_csv(model_name):
    csv_file = "SentimentAssignmentReviewCorpus.csv"  # Replace with the actual path to your CSV file
    df = pd.read_csv(csv_file)

    if not {'reviewTitle', 'reviewBody'}.issubset(df.columns):
        raise ValueError("CSV must contain 'ReviewTitle' and 'ReviewBody' columns")

    results = []

    # Load models if using Hugging Face
    if model_name != "groq":
        aspect_extractor, sentiment_analyzer = load_aspect_model()

    chat_history = []

    for index, row in df.iterrows():
        review_title = row['reviewTitle']
        review_body = row['reviewBody']
        review_text = f"{review_title}. {review_body}"

        print(f"Processing review {index + 1}/{len(df)}")

        if model_name == "groq":
            response = get_groq_aspect_sentiment(review_text, chat_history)
            results.append({
                "ReviewTitle": review_title,
                "ReviewBody": review_body,
                "Aspects": response.get("Aspects", {}),
                "Sentiment": response.get("Sentiment", "N/A"),
                "GeneratedResponse": response.get("GeneratedResponse", ""),
            })
        else:
            response = get_huggingface_aspect_sentiment(review_text, aspect_extractor, sentiment_analyzer)
            results.append({
                "ReviewTitle": review_title,
                "ReviewBody": review_body,
                "Aspects": response.get("Aspects", {}),
                "Sentiment": response.get("Sentiment", "N/A"),
                "Confidence": response.get("Confidence", 0),
                "GeneratedResponse": response.get("GeneratedResponse", ""),
                "Error": response.get("Error", "")
            })

    result_df = pd.DataFrame(results)
    result_csv = "review_aspect_sentiment_analysis_with_responses.csv"
    result_df.to_csv(result_csv, index=False)

    print(f"Results saved to {result_csv}")

# Chat function to interact with the user
def chat(model_name):
    if model_name == "groq":
        print("Chatbot: Using Groq model.")
    else:
        # Load aspect and sentiment models for Hugging Face
        aspect_extractor, sentiment_analyzer = load_aspect_model()
        print("Chatbot: Using Hugging Face model with Aspect-Based Sentiment Analysis.")

    chat_history = []
    print("Chatbot: Hello! You can switch models at any time. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        if model_name == "groq":
            bot_response = get_groq_aspect_sentiment(user_input, chat_history)
        else:
            bot_response = get_huggingface_aspect_sentiment(user_input, aspect_extractor, sentiment_analyzer)

        print("Chatbot:", bot_response)

        # Update chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    # Get model choice from command line arguments or prompt user
    if len(sys.argv) > 1 and sys.argv[1] in llm_options:
        model_choice = llm_options[sys.argv[1]]
    else:
        print("Available models:")
        for key in llm_options:
            print(f"- {key}")
        model_choice = input("Select a model: ").strip().lower()
        if model_choice not in llm_options:
            print("Invalid model selected. Defaulting to Groq.")
            model_choice = "groq"
        else:
            model_choice = llm_options[model_choice]

    # Process the reviews from the hardcoded CSV file
    process_reviews_from_csv(model_choice)
