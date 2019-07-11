import os

__title__ = 'bertsum'
__version__ = os.environ["packageVersion"] if "packageVersion" in os.environ else "0.0.1"
__uri__ = 'https://arxiv.org/pdf/1903.10318.pdf'
__description__ = 'Fine-tune BERT for Extractive Summarization'
