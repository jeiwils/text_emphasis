# Textual Emphasis Analysis

A Python package for analyzing textual emphasis through linguistic and network-based approaches.

## Features

- Text preprocessing (tokenization, cleaning, normalization)
- Embeddings and concept extraction
- Lexical metrics analysis
- Grammatical/syntactic metrics
- Network/structural metrics
- Integration and analysis tools

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Install spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```
textual-emphasis/
├── src/
│   └── textual_emphasis/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── embeddings.py
│       ├── lexical.py
│       ├── syntactic.py
│       ├── network.py
│       └── integration.py
├── tests/
├── notebooks/
├── requirements.txt
├── setup.py
└── README.md
```

## Usage

Example usage will be provided as the package develops.

## Development

To set up the development environment:

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in editable mode:
```bash
pip install -e .
```

## License

MIT License