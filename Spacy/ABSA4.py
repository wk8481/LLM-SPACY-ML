import io
import spacy
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from textblob import TextBlob
from tqdm import tqdm

# Load FastText vectors
def load_vectors(fname):
    with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))
    return data

# Function to read words from a file and return a set for faster lookups
def load_words_from_file(file_name):
    with open(file_name, 'r') as file:
        return set(line.strip() for line in file)

# Load positive, negative, and neutral words from external files
positive_words_file = '/home/zamlamb/KdG/Data n Ai 6/week 3/spacy/ABSA/positive_words.txt'
negative_words_file = '/home/zamlamb/KdG/Data n Ai 6/week 3/spacy/ABSA/negative_words.txt'
neutral_words_file = '/home/zamlamb/KdG/Data n Ai 6/week 3/spacy/ABSA/negation_words.txt'

positive_words = load_words_from_file(positive_words_file)
negative_words = load_words_from_file(negative_words_file)
neutral_words = load_words_from_file(neutral_words_file)

# Load FastText vectors
fasttext_vectors = load_vectors('/home/zamlamb/KdG/Data n Ai 6/week 3/spacy/ABSA/wiki-news-300d-1M-subword.vec')

# Load the CSV file
df = pd.read_csv("SentimentAssignmentReviewCorpus.csv")

# Fill NaN values
df['reviewTitle'] = df['reviewTitle'].fillna("No title provided").astype(str)
df['reviewBody'] = df['reviewBody'].fillna("No review provided").astype(str)

# Load SpaCy models for English and Spanish
nlp_en = spacy.load('en_core_web_sm')
nlp_es = spacy.load('es_core_news_sm')

# Expanded sentiment lexicon with additional multiword phrases and words
sentiment_lexicon = {
    'positive': list(positive_words) + [
        'love', 'great', 'awesome', 'perfect', 'comfortable',
        'easy', 'beautiful', 'fantastic', 'wonderful',
        'excellent', 'satisfied', 'enjoy', 'recommend',
        'happy', 'delighted', 'pleasant', 'impressive',
        'not bad', 'highly recommend', 'well done', 'very good',
        'amazing', 'superb', 'outstanding', 'exceptional',
        'best', 'brilliant', 'remarkable', 'super', 'terrific',
        'fabulous', 'marvelous', 'splendid', 'first-rate',
        'top-notch', 'first-class', 'high quality', 'five star',
        'thumbs up', 'two thumbs up', 'worth every penny'
    ],
    'negative': list(negative_words) + [
        'bad', 'worst', 'waste', 'annoying', 'expensive',
        'uncomfortable', 'terrible', 'horrible', 'not worth it',
        'poor', 'awful', 'dreadful', 'abysmal', 'atrocious',
        'disappointing', 'subpar', 'inferior', 'lousy', 'crappy',
        'pathetic', 'useless', 'worthless', 'garbage', 'junk',
        'rip off', 'not good', 'very bad', 'poor quality',
        'one star', 'thumbs down', 'not recommended'
    ],
    'neutral': list(neutral_words)
}

# Function to calculate sentiment score with aspect consideration
def get_sentiment_score(doc):
    pos_score = 0
    neg_score = 0
    sentiment_expressions = []
    aspects = []

    for token in doc:
        lemma = token.lemma_.lower()

        if lemma in sentiment_lexicon['positive']:
            if any(child.dep_ == 'neg' for child in token.children):
                neg_score += 1
                sentiment_expressions.append({'text': token.text, 'sentiment': 'negative'})
            else:
                pos_score += 1
                sentiment_expressions.append({'text': token.text, 'sentiment': 'positive'})
            aspects.append(token.head.text)

        elif lemma in sentiment_lexicon['negative']:
            if any(child.dep_ == 'neg' for child in token.children):
                pos_score += 1
                sentiment_expressions.append({'text': token.text, 'sentiment': 'positive'})
            else:
                neg_score += 1
                sentiment_expressions.append({'text': token.text, 'sentiment': 'negative'})
            aspects.append(token.head.text)

        # Use FastText vectors to enhance sentiment scoring
        if lemma in fasttext_vectors:
            vector = fasttext_vectors[lemma]
            if sum(vector) > 0:
                pos_score += 1
            else:
                neg_score += 1

    # Return overall sentiment based on scores
    if pos_score > neg_score:
        return ('positive', sentiment_expressions, aspects)
    elif neg_score > pos_score:
        return ('negative', sentiment_expressions, aspects)
    else:
        return ('neutral', sentiment_expressions, aspects)

# Safe function to apply sentiment analysis with improved error handling
def safe_get_sentiment(text, language='en'):
    try:
        doc = nlp_en(text) if language == "en" else nlp_es(text)
        return get_sentiment_score(doc)
    except Exception as e:
        print(f"Error processing text: {text}\nException: {e}")
        return ('neutral', [], [])

# Annotate the golden set with true labels
golden_set_data = [
    ("Love the quality", "2nd time purchased from this store", "positive"),
    ("Nice", "Very comfortable", "positive"),
    ("If they actually worked fully....", "Works okay. Little pricey for them though.", "neutral"),
    ("More shedding than my dog!!", "I wanted to like the bed and at first it worked great, it fit perfectly in my dogs crate and he seemed to be comfortable enough with it. HOWEVER, it sheds like CRAZY! I had read a review talking about that, but I decided to buy it anyways cus it could’ve just been their bed. Turns out it’s mine too. Every single day I would have to vacuum inside and around on the outside of his crate because the fur balls that would come off the bed were so much that they would get carried out around the living room (where the crates at) by the normal inside air. It was so annoying. Not to mention my dog has allergies and the amount of fur that came off the bed was very uncomfortable for him to the point he couldn’t sleep in his crate with that bed. It was a problem for me because his crate doesn’t have a base (it was a hand-me-down crate) so the bed had to be there or he would be even more uncomfortable. Anyway, point is this bed although seemed to work for the price, now has to been thrown away because it is completely useless to me and no amount of vacuuming stops the fur from coming off. I imagine the bed would have to be furless for that to happen. So waste of money there.", "negative"),
    ("love this cup", "Perfect for cup of tea", "positive"),
    ("Colorful and looks nice on my front porch.", "Perfect for what I needed.", "positive"),
    ("Size", "Great for small kitchens, highly effective (:", "positive"),
    ("Great hat", "Great hat", "positive"),
    ("Overall a beautiful puzzle", "Love him!! Been waiting to get him for a while and now he's finally part of my collection. I say it's easy to assemble because of experience, if you're a first-timer for the Crystal Puzzles he might be a little easy to do, just depends. Arrived within two days of shipping and he's a beautiful light ocean blue, stands out from a lot of the main colored puzzles I have.", "positive"),
    ("For everyone who loves maple syrup", "Love the flavor. This is the first real great tasting sugar-free maple syrup that I have ever bought.", "positive")
]

golden_set = pd.DataFrame(golden_set_data, columns=['reviewTitle', 'reviewBody', 'true_labels'])

# Perform sentiment and aspect analysis on the golden set
golden_set_results_spacy = []
golden_set_results_textblob = []
golden_set_aspects = []

for index, row in tqdm(golden_set.iterrows(), total=golden_set.shape[0], desc="Processing Golden Set"):
    sentiment_spacy, details, aspects = safe_get_sentiment(row['reviewBody'])
    sentiment_textblob = TextBlob(row['reviewBody']).sentiment.polarity
    sentiment_textblob = 'positive' if sentiment_textblob > 0 else 'negative' if sentiment_textblob < 0 else 'neutral'
    golden_set_results_spacy.append(sentiment_spacy)
    golden_set_results_textblob.append(sentiment_textblob)
    golden_set_aspects.append(aspects)

# Assign results back to golden set DataFrame
golden_set['predicted_sentiment_spacy'] = golden_set_results_spacy
golden_set['predicted_sentiment_textblob'] = golden_set_results_textblob
golden_set['aspects'] = golden_set_aspects

# Calculate metrics for SpaCy
precision_spacy = precision_score(golden_set['true_labels'], golden_set['predicted_sentiment_spacy'], average='weighted')
recall_spacy = recall_score(golden_set['true_labels'], golden_set['predicted_sentiment_spacy'], average='weighted')
f1_spacy = f1_score(golden_set['true_labels'], golden_set['predicted_sentiment_spacy'], average='weighted')

# Calculate metrics for TextBlob
precision_textblob = precision_score(golden_set['true_labels'], golden_set['predicted_sentiment_textblob'], average='weighted')
recall_textblob = recall_score(golden_set['true_labels'], golden_set['predicted_sentiment_textblob'], average='weighted')
f1_textblob = f1_score(golden_set['true_labels'], golden_set['predicted_sentiment_textblob'], average='weighted')

# Print metrics
print("Benchmark Results:")
print(f"SpaCy - Precision: {precision_spacy:.2f}, Recall: {recall_spacy:.2f}, F1 Score: {f1_spacy:.2f}")
print(f"TextBlob - Precision: {precision_textblob:.2f}, Recall: {recall_textblob:.2f}, F1 Score: {f1_textblob:.2f}")

# Save the golden set to a CSV file
golden_set.to_csv('golden_set_reviews.csv', index=False)

# Perform sentiment and aspect analysis on the entire DataFrame
df_results_spacy = []
df_aspects = []

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Reviews"):
    sentiment_spacy, details, aspects = safe_get_sentiment(row['reviewBody'])
    df_results_spacy.append(sentiment_spacy)
    df_aspects.append(aspects)

# Assign results back to DataFrame
df['final_sentiment'] = df_results_spacy
df['aspects'] = df_aspects

# Implement KMeans clustering on the sentiments
def perform_clustering(sentiments):
    vectorized_sentiments = [[1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0] for sentiment in sentiments]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(vectorized_sentiments)
    return kmeans.labels_

# Perform clustering
df['cluster'] = perform_clustering(df['final_sentiment'])

# Save results to CSV including aspects
df.to_csv('sentiment_analysis_results.csv', columns=['reviewTitle', 'reviewBody', 'final_sentiment', 'cluster', 'aspects'], index=False)