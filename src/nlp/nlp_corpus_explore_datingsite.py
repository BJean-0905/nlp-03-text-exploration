"""nlp_corpus_explore.py - Module 3 Script.

Author: Britany Goucher

Date: 2026-11

Purpose

  Perform exploratory analysis of a small, controlled text corpus.
  Demonstrate how structure emerges from token distributions,
  category comparisons, and co-occurrence patterns.

Analytical Questions

- What tokens dominate each category?
- How do categories differ in vocabulary?
- What words appear in similar contexts?
- What structure is visible before using any models?

Notes:
- This module focuses on exploratory analysis (EDA), not modeling.
- Results here prepare for later work with pipelines and embeddings.

Run from root project folder with:

  uv run python -m nlp.nlp_corpus_explore_datingsite
"""

# ============================================================
# Section 1. Setup and Imports
# ============================================================

from collections import defaultdict
import logging
from pathlib import Path

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
import polars as pl

print("Imports complete.")


# ============================================================
# Configure Logging
# ============================================================

LOG: logging.Logger = get_logger("CI", level="DEBUG")

ROOT_PATH: Path = Path.cwd()
DATA_PATH: Path = ROOT_PATH / "Data"
SCRIPTS_PATH: Path = ROOT_PATH / "scripts"

log_header(LOG, "MODULE 3: CORPUS EXPLORATION")
LOG.info("START script.....")

log_path(LOG, "ROOT_PATH", ROOT_PATH)
log_path(LOG, "DATA_PATH", DATA_PATH)
log_path(LOG, "SCRIPTS_PATH", SCRIPTS_PATH)

LOG.info("Logger configured.")

# ============================================================
# Section 2. Define Corpus (Labeled Text Documents)
# ============================================================

# A corpus is a collection of documents.
# Each document includes a category and text.

csv_file = "Data/Dating_site.csv"

profiles_df = pl.read_csv(csv_file)

print(profiles_df.head())
print(profiles_df.columns)

print("\nUnique Signs:")
print(profiles_df["sign"].unique())

corpus = []

for row in profiles_df.iter_rows(named=True):
    corpus.append({"category": row["sign"], "text": row["Self Description"]})

print(f"Corpus contains {len(corpus)} profiles.")

# ============================================================
# Section 3. Examine Dataset Composition
# ============================================================

sign_counts_df: pl.DataFrame = (
    profiles_df.group_by("sign").len().sort(by="len", descending=True)
)

print("\nProfiles per Sign:")
print(sign_counts_df)

# ============================================================
# Section 4. Tokenize and Clean Text
# ============================================================

# Tokenization splits text into word-like units.


# Define a function to tokenize text by lowercasing, splitting on whitespace,
# and stripping common punctuation. We also filter out very short tokens (length <= 2).
# This simple tokenizer is sufficient for our small, controlled corpus.
# Use the string strip() method to remove punctuation from the beginning and end of each token.
# Remove stop words

STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "with",
    "from",
    "about",
    "that",
    "this",
    "these",
    "those",
    "have",
    "has",
    "had",
    "you",
    "your",
    "you're",
    "me",
    "my",
    "myself",
    "i",
    "i'm",
    "i've",
    "i'd",
    "it",
    "it's",
    "was",
    "were",
    "are",
    "am",
    "be",
    "been",
    "being",
    "just",
    "very",
    "really",
    "like",
    "love",
    "but",
    "not",
    "can",
    "will",
    "would",
    "some",
    "more",
}


def tokenize(text: str) -> list[str]:
    """Tokenize the input text into a list of cleaned tokens."""
    tokens = text.lower().split()
    tokens = [t.strip(".,:;!?()[]\"'") for t in tokens if len(t) > 2]
    return [t for t in tokens if t not in STOPWORDS]


def tokenize_phrases(text: str) -> list[str]:
    """Tokenize text for phrase analysis without removing stopwords."""
    tokens = text.lower().split()

    return [t.strip(".,:;!?()[]\"'") for t in tokens if len(t) > 2]


# Define a new empty list to hold the token records we will create.
records_list: list[dict[str, str]] = []

# Loop through each document, tokenize the text,
# and create a record for each token with its category and
# add it to our list of records.
for doc in corpus:
    # Call our function to tokenize the text of the current document.
    tokens = tokenize(doc["text"])
    # Loop through each token produced by the tokenizer and
    # create a record that includes the category of the document and the token itself.
    # Append this record to our list of records.
    for token in tokens:
        records_list.append({"category": doc["category"], "token": token})

# Create a Polars DataFrame from the list of token records for easier analysis.
token_df: pl.DataFrame = pl.DataFrame(records_list)

# Show results
print("Tokenization complete.")
print(token_df.head(10))


# ============================================================
# Section 5. Compute Global Token Frequencies
# ============================================================

# Frequency distribution = how often each token appears.

# Create a DataFrame that groups the tokens by their text and
# counts how many times each token appears across the entire corpus.
global_freq_df: pl.DataFrame = (
    token_df.group_by("token").len().sort("len", descending=True)
)

# Show results
print("Top global tokens:")
print(global_freq_df.head(10))


# ============================================================
# Section 6. Compute Token Frequencies by Category
# ============================================================

# Compare token usage across categories.

# Create a new DataFrame that groups the tokens by both their category and text,
# counts how many times each token appears within each category,
# and sorts the results first by category and then by frequency in descending order.
# This shows which tokens are most common within each category.
category_freq_df: pl.DataFrame = (
    token_df.group_by(["category", "token"])
    .len()
    .sort(["category", "len"], descending=True)
)

# Show results
print("Top tokens by category:")
print(category_freq_df.head(12))


# ============================================================
# Section 7. Identify Top Tokens per Category
# ============================================================

# Show top tokens per category.


# Define a new empty dictionary to store the top tokens for each category.
top_per_category_dict: dict[str, list[str]] = {}

# Loop through each unique category in the token DataFrame,
# filter the category frequency DataFrame to get the top 5 tokens for that category,
# and store the list of top tokens in the dictionary.
# Also, print the top tokens for each category.
for category in token_df["category"].unique().to_list():
    subset_df = category_freq_df.filter(pl.col("category") == category).head(10)
    top_tokens_list = subset_df["token"].to_list()
    top_per_category_dict[category] = top_tokens_list

    # Show results for this category
    print(f"\n{category.upper()} TOP TOKENS")
    print(top_tokens_list)


# ============================================================
# Section 8. Analyze Co-occurrence (Context Windows)
# ============================================================

# Co-occurrence examines which tokens appear near each other.

# Define how many tokens on each side of a target token we include as context.
# A window size of 2 means:
#   - up to 2 tokens before the target token
#   - up to 2 tokens after the target token
# The target token itself is not included in its context list.
WINDOW_SIZE: int = 2

# Define a new empty dictionary to store the co-occurrence information.
# The keys will be target tokens,
# and the values will be lists of context tokens that appear near the target token.
co_occurrence_dict: dict[str, list[str]] = defaultdict(list)

# Loop through each document in the corpus, tokenize the text,
# and for each token, determine its context tokens based on the defined window size.
for doc in corpus:
    tokens = tokenize(doc["text"])
    for i, token in enumerate(tokens):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)
        context = tokens[start:end]
        for ctx in context:
            if ctx != token:
                co_occurrence_dict[token].append(ctx)

# Show results

sample_tokens = list(co_occurrence_dict.keys())[:5]

for target in sample_tokens:
    print(f"\nContext for '{target}':")
    print(co_occurrence_dict[target][:10])

# ============================================================
# Section 9. Create Bigrams (Local Word Pairs) and Compute Frequencies
# ============================================================

# Bigrams combine each word with the next word in the text.
# This helps us capture local structure: how words are used together,
# not just which words appear individually.

# Bigrams capture pairs of consecutive tokens.

# Define a new empty list to hold the bigram tuples we will create.
bigrams_list: list[tuple[str, str]] = []

# Loop through each document in the corpus, tokenize the text,
# and create bigrams by pairing each token with the next token in the list.
for doc in corpus:
    tokens = tokenize(doc["text"])
    for i in range(len(tokens) - 1):
        bigrams_list.append((tokens[i], tokens[i + 1]))

# Create a DataFrame from the list of bigram tuples,
# where each bigram is represented as a single string with the two tokens separated by a space.
bigram_df: pl.DataFrame = pl.DataFrame(
    {"bigram": [f"{a} {b}" for a, b in bigrams_list]}
)

# Create a new DataFrame that groups the bigrams by their text
# and counts how many times each bigram appears,
# then sorts the results by frequency in descending order.
bigram_freq_df: pl.DataFrame = (
    bigram_df.group_by("bigram").len().sort("len", descending=True)
)

# Show results
print("Top bigrams:")
print(bigram_freq_df.head(20))

# ============================================================
# Section 10. Create Trigrams
# ============================================================

trigrams_list = []

for doc in corpus:
    tokens = tokenize_phrases(doc["text"])

    for i in range(len(tokens) - 2):
        trigrams_list.append((tokens[i], tokens[i + 1], tokens[i + 2]))

trigram_df = pl.DataFrame({"trigram": [f"{a} {b} {c}" for a, b, c in trigrams_list]})

trigram_freq_df: pl.DataFrame = (
    trigram_df.group_by("trigram")
    .len()
    .filter(pl.col("len") > 1)
    .sort(by="len", descending=True)
)

print("\nTop Trigrams:")
print(trigram_freq_df.head(20))

# ============================================================
# Section 11. Visualize Token Frequencies
# ============================================================

print("IMPORTANT: Close chart window to continue execution.")

# Select a zodiac sign to visualize
selected_sign = token_df["category"].unique().sort()[0]

sign_df = category_freq_df.filter(pl.col("category") == selected_sign).head(n=10)

# Create a figure that is 8 inches wide and 4 inches tall
plt.figure(figsize=(8, 4))

# Create bar chart
plt.bar(sign_df["token"], sign_df["len"])

# Rotate labels
ax = plt.gca()
ax.tick_params(axis="x", labelrotation=45)

# Titles and labels
plt.title(
    f"Top Tokens Used by {selected_sign} Profiles\n(Close this window to continue)"
)
plt.xlabel("Token")
plt.ylabel("Frequency")

# Adjust layout
plt.tight_layout()

# Show chart
plt.show()


# ============================================================
# Section 12. Interpret Results and Identify Patterns
# ============================================================

print("\nBRITANY SPECIFIC OBSERVATIONS:")

print(
    "Common Dating site themes appeared more clearly in the bigram and trigram analysis than in the single token analysis."
)
print("Stopword removal improved token frequency clarity, but some context was lost.")
print(
    "Multi-word expressions such as 'get along with' and 'being around people' were revealed in the Trigram analysis."
)
print(
    "Sample size is small and limited sign specific conclusions. However, the analysis demonstrateshow structure emerges from token distributions, category comparisons, and co-occurrence patterns."
)

# ============================================================
# END
# ============================================================

LOG.info("========================")
LOG.info("Pipeline executed successfully!")
LOG.info("========================")
LOG.info("END main()")
