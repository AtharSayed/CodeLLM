
import nltk 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.util import ngrams
from collections import Counter

import numpy as np

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('maxent_ne_chunker_tab')

# 1) First tokenizing the data from the text file 

def tokenize_text_file(file_path):
    try:
        with open(file_path,'r') as  file:
            text = file.read()

        tokens = word_tokenize(text)
        return tokens 
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found")

        return []

# 2) Second Removing Stopwords by passsing the tokens as a  parameter to the stopwords function

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    return filtered_words

# 3.1) Applying stemming from the filtered stopwords 

def apply_stemming(filtered_words):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return stemmed_words

# 3.2) Applying Lemmatization by taking filtered words as parameter
def apply_lemmatization(filtered_words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return lemmatized_words


def is_valid_word(word):
    return wordnet.synsets(word)

# 3.3) Comparing  the performance of stemming vs lemmatization:

def compare_stemming_vs_lemmatization(filtered_words):
    stemmed_words = apply_stemming(filtered_words)
    lemmatized_words = apply_lemmatization(filtered_words)
    
    valid_stemmed_words = [word for word in stemmed_words if is_valid_word(word)]
    valid_lemmatized_words = [word for word in lemmatized_words if is_valid_word(word)]
    
    stemmed_valid_percentage = len(valid_stemmed_words) / len(stemmed_words) * 100 if len(stemmed_words) > 0 else 0
    lemmatized_valid_percentage = len(valid_lemmatized_words) / len(lemmatized_words) * 100 if len(lemmatized_words) > 0 else 0

    print(f"Stemmed Words: {stemmed_words}")
    print(f"Valid Stemmed Words: {valid_stemmed_words}")
    print(f"Percentage of valid stemmed words: {stemmed_valid_percentage:.2f}%\n")
    
    print(f"Lemmatized Words: {lemmatized_words}")
    print(f"Valid Lemmatized Words: {valid_lemmatized_words}")
    print(f"Percentage of valid lemmatized words: {lemmatized_valid_percentage:.2f}%\n")
    
    if stemmed_valid_percentage > lemmatized_valid_percentage:
        return "Stemming performed better in terms of valid words."
    elif lemmatized_valid_percentage > stemmed_valid_percentage:
        return "Lemmatization performed better in terms of valid words."
    else:
        return "Both methods performed equally well in terms of valid words."


# 4) Performing   One-Hot Encoding
def one_hot_encoding(filtered_words, vocab):
    one_hot_encoded = []
    for word in filtered_words:
        one_hot_vector = [1 if word == v else 0 for v in vocab]
        one_hot_encoded.append(one_hot_vector)
    return np.array(one_hot_encoded)

# 5) Function for Bag of Words (BoW) representation
def bag_of_words(filtered_words, vocab):
    word_count = [filtered_words.count(word) for word in vocab]
    return np.array(word_count)


# 6) Performing  Named Entity Recognition (NER) by passing the filtered_words as a parameter to the named_entity_recognition function 
def named_entity_recognition(filtered_words):
    tagged_tokens = pos_tag(filtered_words)  
    named_entities = ne_chunk(tagged_tokens)  

    # Extract named entities (i.e., Person, Location, Organization)
    named_entities_list = []
    for tree in named_entities:
        if isinstance(tree, Tree):
            entity = " ".join([word for word, tag in tree.leaves()])
            entity_type = tree.label()
            named_entities_list.append((entity, entity_type))
    
    return named_entities_list

# 7) Performing  POS Tagging and returning  the tags
def pos_tagging(filtered_words):
    tagged_tokens = pos_tag(filtered_words) 
    return tagged_tokens

# 8) Analyzing the  frequency of POS tags
def analyze_pos_frequency(tagged_tokens):
    pos_frequency = nltk.FreqDist(tag for word, tag in tagged_tokens)
    return pos_frequency

# 9) Generating N-Gram
def generate_ngrams(filtered_tokens, n):
    n_grams = ngrams(filtered_tokens, n)  
    return list(n_grams)

# 10) Analyzing the  frequency of N-grams
def analyze_ngram_frequency(ngrams_list):
    # Counting the frequency of each N-gram
    ngram_frequency = Counter(ngrams_list)  
    return ngram_frequency

if __name__=="__main__":

    file_path =r"F:\M.Tech_CollgeMaterials\CodeLLM\PracticalLabs\Lb1\Sample.txt"

    tokens = tokenize_text_file(file_path)

    if tokens:
        print("1.Tokenized text:  ",tokens)
        
        filtered_tokens = remove_stopwords(tokens)
        
        print("\n2.Filtered tokens removed stopwords : ",filtered_tokens)

        stemmed_tokens = apply_stemming(filtered_tokens)
        
        print("\n3.Stemmed Tokens:",stemmed_tokens)

        lemmatized_tokens = apply_lemmatization(filtered_tokens)
        
        print("\n4.Lemmatized Tokens:",lemmatized_tokens)

        comparison_res = compare_stemming_vs_lemmatization(filtered_tokens)

        print("\n5.Final Comparison  stemming vs lemmatization :\n",comparison_res)

        # Create vocabulary (unique words in the filtered tokens)
        vocab = list(set(filtered_tokens))
    
        # One-Hot Encoding
        one_hot_encoded = one_hot_encoding(filtered_tokens, vocab)
        print("\n6. One-Hot Encoded Representation:",one_hot_encoded)
    
        # Bag of Words
        bow_representation = bag_of_words(filtered_tokens, vocab)
        print("\n7. Bag of Words Representation:",bow_representation)

        # Performed NER 
        named_entities = named_entity_recognition(filtered_tokens)
        print("\n8. Named Entities in the  Filtered Words:")
        for entity, entity_type in named_entities:
            print(f"{entity} ({entity_type})")
            
        tagged_tokens = pos_tagging(filtered_tokens)
        print("\n9.POS Tagged Tokens:")
        print(tagged_tokens)

        pos_frequency = analyze_pos_frequency(tagged_tokens)
        print("\n10.POS Tag Frequency Analysis:",pos_frequency)

        # Generate bi-grams and tri-grams
        bi_grams = generate_ngrams(filtered_tokens, 2)
        tri_grams = generate_ngrams(filtered_tokens, 3)

        # Analyze frequency of bi-grams and tri-grams
        bi_gram_frequency = analyze_ngram_frequency(bi_grams)
        tri_gram_frequency = analyze_ngram_frequency(tri_grams)

        # Print the most common bi-grams and tri-grams
        print("\n11.Most Common Bi-grams:")
        for ngram, freq in bi_gram_frequency.most_common(5):
            print(f"{ngram}: {freq}")

        print("\n12.Most Common Tri-grams:")
        for ngram, freq in tri_gram_frequency.most_common(5):
            print(f"{ngram}: {freq}")

