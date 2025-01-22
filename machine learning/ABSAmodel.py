
import pandas as pd
import nltk
from gitdb.fun import chunk_size
from nltk.corpus import words, wordnet, stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc

# Ignore the transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Download tokenizer model which is used for tokeninzing sentences into words
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')

# load text
text = pd.read_csv("SentimentAssignmentReviewCorpus.csv", sep=",")

# Load ABSA pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

# Load the sentiment lexicon from file
def load_wards_from_file(file_path):
    with open(file_path, 'r') as file:
        return set(file.read().splitlines())

positive_words_file = r'C:\Users\mnmhy\IntelliJprojects\DAI6_LLM\positive_words.txt'
negative_words_file = r'C:\Users\mnmhy\IntelliJprojects\DAI6_LLM\negative_words.txt'
neutral_words_file = r'C:\Users\mnmhy\IntelliJprojects\DAI6_LLM\negation_words.txt'

positive_words = load_wards_from_file(positive_words_file)
negative_words = load_wards_from_file(negative_words_file)
neutral_words = load_wards_from_file(neutral_words_file)

sentiment_lexicon = {
    'positive': positive_words,
    'negative': negative_words,
    'neutral': neutral_words
}

# Tokenizes the review and POS tagging, then extracts the noun phrases and named entities as aspects of the review
def extract_aspects(review):
    tokens = nltk.word_tokenize(review)
    tagged = nltk.pos_tag(tokens)
    aspects = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return aspects

# Function to tokenize a review and check for multiword expressions
# if a multiword expression is found, return the sentiment value
# if "not" is followed by a word, it returns the negative sentiment value of the word
def handle_negation_and_multiword(review, lexicon):
    tokens = nltk.word_tokenize(review)
    for i in range(len(tokens) - 1):
        phrase = tokens[i].lower() + " " + tokens[i + 1].lower()
        if phrase in lexicon:
            return lexicon[phrase]  # Return the multiword sentiment if found
        if tokens[i].lower() == 'not' and tokens[i + 1].lower() in lexicon:
            return -lexicon[tokens[i + 1].lower()]  # Handle negation
    return None

# Function to convert the review to lowercase and tokenize, perform sentiment analysis using the ABSA model
def analyze_review_absa_with_aspects(reviews):
    reviews = [review.lower() for review in reviews]
    tokens_list = [nltk.word_tokenize(review) for review in reviews]

    # Step 1: Check for multiword expressions and negations
    sentiments_from_lexicon = [handle_negation_and_multiword(review, sentiment_lexicon) for review in reviews]

    # Step 2: If sentiment is found in the lexicon, return it
    results = []
    for i, sentiment in enumerate(sentiments_from_lexicon):
        if sentiment is not None:
            results.append({
                "overall_score": sentiment,
                "aspects": extract_aspects(reviews[i]),
                "sentiments": []
            })
        else:
            results.append(None)

    # Step 3: If no lexicon match, use the ABSA model to predict
    non_lexicon_indices = [i for i in range(len(reviews)) if results[i] is None]
    non_lexicon_reviews = [reviews[i] for i in non_lexicon_indices]

    if non_lexicon_reviews:  # Only proceed if there are reviews that need ABSA prediction
        inputs = tokenizer(non_lexicon_reviews, return_tensors="pt", truncation=True, max_length=128, padding=True)
        outputs = model(**inputs)
        sentiment_scores = F.softmax(outputs.logits, dim=1)

        sentiments = ["negative", "neutral", "positive"]
        predicted_sentiments = [sentiments[scores.argmax()] for scores in sentiment_scores]

        # Step 4: Extract aspects for ABSA-predicted reviews and update results
        for idx, review in enumerate(non_lexicon_reviews):
            aspects = extract_aspects(review)
            results[non_lexicon_indices[idx]] = {
                "overall_score": predicted_sentiments[idx],
                "aspects": aspects,
                "sentiments": []
            }

    # Step 5: Identify aspects
    for i, tokens in enumerate(tokens_list):
        for word in tokens:
            if isinstance(word, str) and (word in sentiment_lexicon['positive'] or word in sentiment_lexicon['negative']):
                sentiment_value = "positive" if word in sentiment_lexicon['positive'] else "negative"
                aspect = {
                    "word": word,
                    "sentiment": sentiment_value
                }
                results[i]["sentiments"].append(aspect)

    return results

# Handle missing data
text['reviewTitle'] = text['reviewTitle'].fillna('')
text['reviewBody'] = text['reviewBody'].fillna('')

# Function to process reviews in parallel
# def process_reviews_in_parallel(reviews, func, max_workers=4, batch_size=32):
#     results = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for i in tqdm(range(0, len(reviews), batch_size), desc="Processing reviews"):
#             batch = reviews[i:i + batch_size]
#             results.extend(list(executor.map(func, batch)))
#             gc.collect() # Clear memory after each batch
#     return results
#


# Processing
text['title_result'] = text['reviewTitle'].apply(lambda review: analyze_review_absa_with_aspects([review]))
text['body_result'] = text['reviewBody'].apply(lambda review: analyze_review_absa_with_aspects([review]))

# Extract aspect
text['aspects'] = text.apply(lambda row: (row['title_result'][0]['aspects'] if row['title_result'] and row['title_result'][0] else []) +
                                      (row['body_result'][0]['aspects'] if row['body_result'] and row['body_result'][0] else []), axis=1)

# Save the results to a CSV file
text.to_csv("absa_analysis_results.csv", index=False)

# Show the first few results
print(text[['reviewTitle', 'title_result', 'reviewBody', 'body_result']].head())