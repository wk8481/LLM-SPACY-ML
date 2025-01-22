# LLM-SPACY-ML

This repository contains various resources, code, and documentation related to sentiment analysis using Large Language Models (LLMs), SpaCy, and machine learning algorithms. It includes slide decks, data files, Python scripts, and documentation to help you understand and implement sentiment analysis techniques.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Tools](#models-and-tools)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The goal of this repository is to provide a comprehensive set of tools and resources for performing sentiment analysis using various techniques. It includes implementations using SpaCy, LLMs, and traditional machine learning algorithms. The repository also contains documentation and slide decks to help you understand the concepts and implementations.

## Repository Structure

```
LLM-SPACY-ML/
├── Spacy/
│   ├── ABSA4.py
│   ├── positive_words.txt
│   ├── negative_words.txt
│   ├── negation_words.txt
│   ├── golden_set_reviews.csv
│   ├── sentiment_analysis_results.csv
│   └── F1 scores.docx
├── llm sentiment/
│   ├── llm_sentiment_analysis.csv
│   ├── sentimentAnalysisFeedback.py
│   ├── review_aspect_sentiment_analysis_with_responses.csv
│   ├── review_aspect_sentiment_analysis_with_responses.xlsx
├── machine learning/
│   ├── ABSAmodel.py
│   ├── absa_analysis_results.csv
├── mllm/
│   ├── llavareal.py
│   ├── photo of mllm.docx
├── Slide deck/
│   ├── LLMs Use Cases.pptx
│   ├── MLLMs Use Cases.pptx
├── Sentiment Analysis Comparisons.docx
├── Week 2 LLM Project.docx
├── requirements.txt
└── README.md
```

### Key Files and Directories

- **Spacy/**: Contains scripts and data files for sentiment analysis using SpaCy.
  - `ABSA4.py`: Script for Aspect-Based Sentiment Analysis (ABSA) using SpaCy.
  - `positive_words.txt`, `negative_words.txt`, `negation_words.txt`: Lexicon files for sentiment analysis.
  - `golden_set_reviews.csv`, `sentiment_analysis_results.csv`: CSV files with sentiment analysis results.
  - `F1 scores.docx`: Document with F1 scores for the sentiment analysis models.

- **llm sentiment/**: Contains scripts and data files for sentiment analysis using Large Language Models (LLMs).
  - `llm_sentiment_analysis.csv`: CSV file with sentiment analysis results using LLMs.
  - `sentimentAnalysisFeedback.py`: Script for generating sentiment analysis feedback using LLMs.
  - `review_aspect_sentiment_analysis_with_responses.csv`, `review_aspect_sentiment_analysis_with_responses.xlsx`: Files with detailed sentiment analysis results and responses.

- **machine learning/**: Contains scripts and data files for sentiment analysis using traditional machine learning algorithms.
  - `ABSAmodel.py`: Script for Aspect-Based Sentiment Analysis (ABSA) using machine learning models.
  - `absa_analysis_results.csv`: CSV file with sentiment analysis results using machine learning models.

- **mllm/**: Contains scripts and documentation related to Multi-Modal Language Models (MLLMs).
  - `llavareal.py`: Script for image analysis using MLLMs.
  - `photo of mllm.docx`: Document with information about MLLMs.

- **Slide deck/**: Contains slide decks with use cases and explanations of LLMs and MLLMs.
  - `LLMs Use Cases.pptx`: Slide deck with use cases for LLMs.
  - `MLLMs Use Cases.pptx`: Slide deck with use cases for MLLMs.

- **Sentiment Analysis Comparisons.docx**: Document comparing different sentiment analysis methods.

- **Week 2 LLM Project.docx**: Documentation for a project on LLMs.

- **requirements.txt**: File listing the Python dependencies required for the project.

## Installation

To use the scripts and resources in this repository, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wk8481/LLM-SPACY-ML.git
   cd LLM-SPACY-ML
   ```

2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the SpaCy Sentiment Analysis

1. Navigate to the `Spacy` directory:
   ```bash
   cd Spacy
   ```

2. Run the `ABSA4.py` script:
   ```bash
   python ABSA4.py
   ```

### Running the LLM Sentiment Analysis

1. Navigate to the `llm sentiment` directory:
   ```bash
   cd llm sentiment
   ```

2. Run the `sentimentAnalysisFeedback.py` script:
   ```bash
   python sentimentAnalysisFeedback.py
   ```

### Running the Machine Learning Sentiment Analysis

1. Navigate to the `machine learning` directory:
   ```bash
   cd machine learning
   ```

2. Run the `ABSAmodel.py` script:
   ```bash
   python ABSAmodel.py
   ```

### Running the MLLM Image Analysis

1. Navigate to the `mllm` directory:
   ```bash
   cd mllm
   ```

2. Run the `llavareal.py` script using Streamlit:
   ```bash
   streamlit run llavareal.py
   ```

## Models and Tools

- **LLM Models**: Various Large Language Models, including GPT-2, GPT-Neo, and others, used for sentiment analysis.
- **SpaCy**: A popular NLP library used for text processing and sentiment analysis.
- **Machine Learning**: Traditional machine learning algorithms used for Aspect-Based Sentiment Analysis (ABSA).
- **MLLM**: Multi-Modal Language Models used for image analysis and other multi-modal tasks.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes. Make sure to follow the repository's coding standards and guidelines.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or inquiries, please contact [williamkasasa26@gmail.com](mailto:williamkasasa26@gmail.com).
