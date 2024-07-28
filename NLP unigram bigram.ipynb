{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "19aa3245-5328-4dc3-be09-e953f56792c1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unigrams: Counter({'this': 2, 'is': 2, 'text': 2, 'corpus': 2, '.': 2, 'a': 1, 'sample': 1, 'used': 1, 'to': 1, 'demonstrate': 1, 'processing': 1})\n",
      "\n",
      "Bigrams: Counter({('this', 'is'): 1, ('is', 'a'): 1, ('a', 'sample'): 1, ('sample', 'text'): 1, ('text', 'corpus'): 1, ('corpus', '.'): 1, ('.', 'this'): 1, ('this', 'corpus'): 1, ('corpus', 'is'): 1, ('is', 'used'): 1, ('used', 'to'): 1, ('to', 'demonstrate'): 1, ('demonstrate', 'text'): 1, ('text', 'processing'): 1, ('processing', '.'): 1})\n",
      "\n",
      "Trigrams: Counter({('this', 'is', 'a'): 1, ('is', 'a', 'sample'): 1, ('a', 'sample', 'text'): 1, ('sample', 'text', 'corpus'): 1, ('text', 'corpus', '.'): 1, ('corpus', '.', 'this'): 1, ('.', 'this', 'corpus'): 1, ('this', 'corpus', 'is'): 1, ('corpus', 'is', 'used'): 1, ('is', 'used', 'to'): 1, ('used', 'to', 'demonstrate'): 1, ('to', 'demonstrate', 'text'): 1, ('demonstrate', 'text', 'processing'): 1, ('text', 'processing', '.'): 1})\n",
      "\n",
      "Bigram Probabilities:\n",
      "P(is|this) = 0.5\n",
      "P(corpus|this) = 0.5\n",
      "P(a|is) = 0.5\n",
      "P(used|is) = 0.5\n",
      "P(sample|a) = 1.0\n",
      "P(text|sample) = 1.0\n",
      "P(corpus|text) = 0.5\n",
      "P(processing|text) = 0.5\n",
      "P(.|corpus) = 0.5\n",
      "P(is|corpus) = 0.5\n",
      "P(this|.) = 1.0\n",
      "P(to|used) = 1.0\n",
      "P(demonstrate|to) = 1.0\n",
      "P(text|demonstrate) = 1.0\n",
      "P(.|processing) = 1.0\n",
      "\n",
      "Next word predictions for 'this': ['is', 'corpus']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\ASUS\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "from collections import Counter, defaultdict\n",
    "import nltk\n",
    "from nltk.util import ngrams\n",
    "import random\n",
    "\n",
    "nltk.download('punkt')\n",
    "\n",
    "# Sample text corpus\n",
    "corpus = \"This is a sample text corpus. This corpus is used to demonstrate text processing.\"\n",
    "\n",
    "# Tokenize the text into words\n",
    "tokens = nltk.word_tokenize(corpus.lower())\n",
    "\n",
    "# Unigrams\n",
    "unigrams = Counter(tokens)\n",
    "print(\"Unigrams:\", unigrams)\n",
    "\n",
    "# Bigrams\n",
    "bigrams = list(ngrams(tokens, 2))\n",
    "bigram_counts = Counter(bigrams)\n",
    "print(\"\\nBigrams:\", bigram_counts)\n",
    "\n",
    "# Trigrams\n",
    "trigrams = list(ngrams(tokens, 3))\n",
    "trigram_counts = Counter(trigrams)\n",
    "print(\"\\nTrigrams:\", trigram_counts)\n",
    "\n",
    "# Bigram probabilities\n",
    "bigram_probabilities = defaultdict(lambda: defaultdict(int))\n",
    "for w1, w2 in bigrams:\n",
    "    bigram_probabilities[w1][w2] += 1\n",
    "\n",
    "for w1 in bigram_probabilities:\n",
    "    total_count = float(sum(bigram_probabilities[w1].values()))\n",
    "    for w2 in bigram_probabilities[w1]:\n",
    "        bigram_probabilities[w1][w2] /= total_count\n",
    "\n",
    "print(\"\\nBigram Probabilities:\")\n",
    "for w1 in bigram_probabilities:\n",
    "    for w2 in bigram_probabilities[w1]:\n",
    "        print(f\"P({w2}|{w1}) = {bigram_probabilities[w1][w2]}\")\n",
    "\n",
    "# Next word prediction function\n",
    "def predict_next_word(word, num_predictions=3):\n",
    "    if word in bigram_probabilities:\n",
    "        sorted_predictions = sorted(bigram_probabilities[word].items(), key=lambda item: item[1], reverse=True)\n",
    "        return [word for word, prob in sorted_predictions[:num_predictions]]\n",
    "    else:\n",
    "        return []\n",
    "\n",
    "# Example: Predict next words for 'this'\n",
    "next_words = predict_next_word('this')\n",
    "print(\"\\nNext word predictions for 'this':\", next_words)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "102a88f4-5d15-4cab-b230-5653832d9e46",
   "metadata": {},
   "outputs": [],
   "source": [
    "##GETTING INPUT FROM USER"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9945dfee-6bf3-447c-9355-b95c6e4bee02",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\ASUS\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Enter a text corpus:  This is a sample text corpus. This corpus is used to demonstrate text processing.This is a sample text corpus. This corpus is used to demonstrate text processing.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Unigrams: Counter({'is': 4, 'text': 4, 'corpus': 4, 'this': 3, '.': 3, 'a': 2, 'sample': 2, 'used': 2, 'to': 2, 'demonstrate': 2, 'processing.this': 1, 'processing': 1})\n",
      "\n",
      "Bigrams: Counter({('is', 'a'): 2, ('a', 'sample'): 2, ('sample', 'text'): 2, ('text', 'corpus'): 2, ('corpus', '.'): 2, ('.', 'this'): 2, ('this', 'corpus'): 2, ('corpus', 'is'): 2, ('is', 'used'): 2, ('used', 'to'): 2, ('to', 'demonstrate'): 2, ('demonstrate', 'text'): 2, ('this', 'is'): 1, ('text', 'processing.this'): 1, ('processing.this', 'is'): 1, ('text', 'processing'): 1, ('processing', '.'): 1})\n",
      "\n",
      "Trigrams: Counter({('is', 'a', 'sample'): 2, ('a', 'sample', 'text'): 2, ('sample', 'text', 'corpus'): 2, ('text', 'corpus', '.'): 2, ('corpus', '.', 'this'): 2, ('.', 'this', 'corpus'): 2, ('this', 'corpus', 'is'): 2, ('corpus', 'is', 'used'): 2, ('is', 'used', 'to'): 2, ('used', 'to', 'demonstrate'): 2, ('to', 'demonstrate', 'text'): 2, ('this', 'is', 'a'): 1, ('demonstrate', 'text', 'processing.this'): 1, ('text', 'processing.this', 'is'): 1, ('processing.this', 'is', 'a'): 1, ('demonstrate', 'text', 'processing'): 1, ('text', 'processing', '.'): 1})\n",
      "\n",
      "Bigram Probabilities:\n",
      "P(is|this) = 0.3333333333333333\n",
      "P(corpus|this) = 0.6666666666666666\n",
      "P(a|is) = 0.5\n",
      "P(used|is) = 0.5\n",
      "P(sample|a) = 1.0\n",
      "P(text|sample) = 1.0\n",
      "P(corpus|text) = 0.5\n",
      "P(processing.this|text) = 0.25\n",
      "P(processing|text) = 0.25\n",
      "P(.|corpus) = 0.5\n",
      "P(is|corpus) = 0.5\n",
      "P(this|.) = 1.0\n",
      "P(to|used) = 1.0\n",
      "P(demonstrate|to) = 1.0\n",
      "P(text|demonstrate) = 1.0\n",
      "P(is|processing.this) = 1.0\n",
      "P(.|processing) = 1.0\n"
     ]
    }
   ],
   "source": [
    "from collections import Counter, defaultdict\n",
    "import nltk\n",
    "from nltk.util import ngrams\n",
    "import random\n",
    "\n",
    "# Download NLTK data if needed\n",
    "nltk.download('punkt')\n",
    "\n",
    "# Get input corpus from the user\n",
    "corpus = input(\"Enter a text corpus: \")\n",
    "\n",
    "# Tokenize the text into words\n",
    "tokens = nltk.word_tokenize(corpus.lower())\n",
    "\n",
    "# Unigrams\n",
    "unigrams = Counter(tokens)\n",
    "print(\"\\nUnigrams:\", unigrams)\n",
    "\n",
    "# Bigrams\n",
    "bigrams = list(ngrams(tokens, 2))\n",
    "bigram_counts = Counter(bigrams)\n",
    "print(\"\\nBigrams:\", bigram_counts)\n",
    "\n",
    "# Trigrams\n",
    "trigrams = list(ngrams(tokens, 3))\n",
    "trigram_counts = Counter(trigrams)\n",
    "print(\"\\nTrigrams:\", trigram_counts)\n",
    "\n",
    "# Bigram probabilities\n",
    "bigram_probabilities = defaultdict(lambda: defaultdict(int))\n",
    "for w1, w2 in bigrams:\n",
    "    bigram_probabilities[w1][w2] += 1\n",
    "\n",
    "for w1 in bigram_probabilities:\n",
    "    total_count = float(sum(bigram_probabilities[w1].values()))\n",
    "    for w2 in bigram_probabilities[w1]:\n",
    "        bigram_probabilities[w1][w2] /= total_count\n",
    "\n",
    "print(\"\\nBigram Probabilities:\")\n",
    "for w1 in bigram_probabilities:\n",
    "    for w2 in bigram_probabilities[w1]:\n",
    "        print(f\"P({w2}|{w1}) = {bigram_probabilities[w1][w2]}\")\n",
    "\n",
    "# Next word prediction function\n",
    "def predict_next_word(word, num_predictions=3):\n",
    "    if word in bigram_probabilities:\n",
    "        sorted_predictions = sorted(bigram_probabilities[word].items(), key=lambda item: item[1], reverse=True)\n",
    "        return [word for word, prob in sorted_predictions[:num_predictions]]\n",
    "    else:\n",
    "        return []\n",
    "\n",
    "# Example: Predict next words for a user-provided word\n",
    "next_word_input = input(\"\\nEnter a word to predict the next word(s): \")\n",
    "next_words = predict_next_word(next_word_input)\n",
    "print(f\"\\nNext word predictions for '{next_word_input}':\", next_words)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de9ec51f-88a7-415b-a82e-2127db8496c2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
