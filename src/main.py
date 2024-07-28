import os
import time
import logging
from datetime import datetime
from collections import Counter
from typing import Dict, List

import fire
from devtopy import DevTo
from dotenv import load_dotenv
import nltk
import numpy as np
from PIL import Image
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Constants
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(THIS_DIR, ".env"))
API_KEY = os.getenv("DEVTO_API_KEY")
MASK_IMG_PATH = os.path.join(THIS_DIR, "mask", "dev_gray.png")
OUTPUT_DIR = os.path.join(THIS_DIR, "output")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_nltk_resources():
    """Download required NLTK resources"""
    for resource in ["punkt", "stopwords", "wordnet"]:
        nltk.download(resource, quiet=True)
    logger.info("NLTK resources downloaded successfully")


def preprocess_text(text: str) -> List[str]:
    """Preprocess text: lowercase, tokenize, remove stopwords and non-alphanumeric tokens"""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]


def lemmatize_words(words: List[str]) -> List[str]:
    """Lemmatize a list of words"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def process_text(text: str) -> str:
    """Process text and return a string of lemmatized words"""
    preprocessed_words = preprocess_text(text)
    lemmatized_words = lemmatize_words(preprocessed_words)
    return " ".join(lemmatized_words)


def create_wordcloud(word_counter: Dict[str, int]) -> None:
    """Generate and save a word cloud image"""
    mask = np.array(Image.open(MASK_IMG_PATH))

    wordcloud = WordCloud(
        width=1920,
        height=1920,
        background_color="white",
        mask=mask,
        contour_width=1,
        contour_color="steelblue",
    ).generate_from_frequencies(word_counter)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    save_wordcloud_image(wordcloud)


def save_wordcloud_image(wordcloud: WordCloud) -> None:
    """Save the word cloud image to a file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"wordcloud_{datetime.now():%Y%m%d%H%M}.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    logger.info(f"WordCloud image saved to: {output_path}")


def fetch_articles(dt: DevTo, article_count: int) -> List[Dict]:
    """Fetch articles from DEV.to"""
    articles = dt.articles.get_latest_articles(page=1, per_page=article_count).articles
    logger.info(f"Fetched {len(articles)} articles from DEV.to")
    return articles


def process_article(dt: DevTo, article: Dict) -> str:
    """Process a single article and return processed text"""
    article_data = dt.articles.get_by_id(article.id)
    time.sleep(1)  # Rate limiting
    logger.debug(f"Processed article: {article.id}")
    return process_text(article_data.body_markdown)


def get_word(article_count: int = 10) -> Counter:
    """Fetch articles from DEV.to and count word occurrences"""
    if not 1 <= article_count <= 1000:
        raise ValueError("article_count must be between 1 and 1000")

    dt = DevTo(api_key=API_KEY)
    articles = fetch_articles(dt, article_count)

    word_counter = Counter()
    for article in tqdm(articles):
        processed_text = process_article(dt, article)
        word_counter.update(processed_text.split())

    logger.info(
        f"Processed {len(articles)} articles and counted {len(word_counter)} unique words"
    )
    return word_counter


def create(article_count: int = 25) -> None:
    """Generate and save a word cloud image"""
    logger.info("Starting the word cloud generation process")
    download_nltk_resources()
    word_counter = get_word(article_count=article_count)
    logger.debug(f"Word counter: {word_counter}")
    create_wordcloud(word_counter)
    logger.info("Word cloud generation process completed")


def main():
    fire.Fire(create)


if __name__ == "__main__":
    main()
