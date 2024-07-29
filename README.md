# DEV.to Word Cloud Generator

This project generates a word cloud image from the latest articles on DEV.to. It fetches articles using the DEV.to API, processes the text, and creates a visually appealing word cloud in the shape of the DEV.to logo.

## Features

- Fetches latest articles from DEV.to
- Processes text using Natural Language Processing techniques
- Generates a word cloud in the shape of the DEV.to logo
- Customizable number of articles to process

## Requirements

- Python 3.7+
- pip

## Installation

1. Clone this repository:

   ```

   git clone <https://github.com/yourusername/devto-word-cloud-generator.git>
   cd devto-word-cloud-generator

   ```

2. Install the required packages:

   ```

   pip install -r requirements.txt

   ```

3. Set up your DEV.to API key:
   - Create a `.env` file in the project root
   - Add your API key: `DEVTO_API_KEY=your_api_key_here`

## Usage

Run the script with the desired number of articles to process:

```

python main.py --article_count=25

```

The default number of articles is 25 if not specified.

## Output

The generated word cloud image will be saved in the `output` directory with a timestamp in the filename.

## Customization

- Modify the `MASK_IMG_PATH` constant to use a different mask image for the word cloud shape.
- Adjust the `WordCloud` parameters in the `create_wordcloud` function to change the appearance of the word cloud.

## Dependencies

- devtopy: A Python library for interacting with the DEV.to API
- NLTK: Natural Language Toolkit for text processing
- WordCloud: For generating the word cloud image
- matplotlib: For plotting and saving the image
- tqdm: For progress bars
- fire: For command-line interface

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
