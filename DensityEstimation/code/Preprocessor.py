from hazm import *
import string
import re
import pandas as pd
import fasttext
from bs4 import BeautifulSoup
import os
from hazm import POSTagger, word_tokenize

class Preprocessor:
    
    def __init__(self):
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()
        self.tagger = POSTagger(model='post_tagger.model')
        self.stop_words = self.get_stop_words()
    
    
    def _remove_invalid_labels(self, dataset):
        valid_label_locations = dataset['label'].isin([0, -1, 1])
        return dataset[valid_label_locations]
    
    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        cleaned_text = text.replace('\u200c', ' ')
        website_pattern = re.compile(r'https?://\S+|www\.\S+')
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        text_without_links = re.sub(website_pattern, ' ', cleaned_text)
        text_without_links = re.sub(email_pattern, ' ', text_without_links)
        number_pattern = re.compile(r'[0-9۰-۹]+')
        number_free_text = re.sub(number_pattern, ' ', text)
        english_pattern = re.compile(r'[a-zA-Z]+')
        english_free_text = re.sub(english_pattern, ' ', number_free_text)
        non_repeated_text = re.sub(r'(.)\1+', r'\1', english_free_text)
        characters_to_remove = ['_', 'М', 'а', 'в', 'к', 'о', 'с']
        cleaned_text = ''.join([char for char in non_repeated_text if char not in characters_to_remove])
        text_special_characters = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in non_repeated_text)
        space_free_text = ' '.join(text_special_characters.split())
        
        return space_free_text

    def _normalization(self, text):
        normalized_text = self.normalizer.normalize(text)
        return normalized_text

    def _tokenization(self, text):
        tokens = word_tokenize(text)
        return tokens

    def get_stop_words(self):
        with open("stop_words.txt", 'r', encoding='utf-8') as file:
            stop_words = set(word.strip() for word in file.readlines())
        return stop_words
    
    def _remove_stop_words(self, tokens):
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens

    def _remove_unnecessary_characters(self, words):
        english_punctuation = string.punctuation
        persian_punctuation = "،؛؟«»"
        all_punctuation = english_punctuation + persian_punctuation
        unnecessary_chars = set(all_punctuation)
        filtered_tokens = [token for token in words if token not in unnecessary_chars]
        return filtered_tokens

    def _stemming(self, token):
        return self.stemmer.stem(token)
        
    def _lemmatizing(self, tokens):
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        def handle_special_token(tokens):
            filtered_tokens = []
            for token in tokens:
                parts = token.split('#')
                if len(parts) == 1:
                    filtered_tokens.append(token)
                    continue
                filtered_tokens.append(parts[1])
            return filtered_tokens
        
        lemmatized_words = handle_special_token(lemmatized_words)
        return lemmatized_words

    def preprocessing(self, dataset):
        dataset = self._remove_invalid_labels(dataset)
        dataset = dataset.dropna(subset=['comment'])
        dataset = dataset[dataset['comment'] != '[" "]']
        def preprocess_comment(comment):
            cleaned_comment = self._clean_text(comment)
            normalized_comment = self._normalization(cleaned_comment)
            tokens = self._tokenization(normalized_comment)
            filtered_tokens = self._remove_stop_words(tokens)
            filtered_tokens = self._remove_unnecessary_characters(filtered_tokens)
#             tags = self.tagger.tag(filtered_tokens)
#             for i, (filtered_token, tag) in enumerate(zip(filtered_tokens, tags)):
#                 if tag[1] == 'VERB':
# #                     print("verb")
#                     stemmed_word = self._stemming(filtered_token)
# #                     print("filtered_token: ", filtered_token)
# #                     lemmatized_word = self._lemmatizing([filtered_token])[0]
# #                     print("stemmed_word: ",stemmed_word)
#                     filtered_tokens[i] = lemmatized_word
            new_comment = ' '.join(filtered_tokens)
            return new_comment, filtered_tokens

        dataset['comment'], dataset['tokens'] = zip(*dataset['comment'].apply(preprocess_comment))
        filtered_df = dataset[dataset['comment'].notna() & (dataset['comment'] != '')]
        return filtered_df
