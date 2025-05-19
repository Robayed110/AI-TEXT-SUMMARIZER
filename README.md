# AI Chat Log Summarizer

This Python tool reads `.txt` chat logs between a user and an AI, and summarizes the conversation by:
- Counting messages from each speaker.
- Extracting the most frequent keywords.
- Optionally applying TF-IDF for better keyword detection.

## Features
- Parses `.txt` chat logs
- Outputs summary of conversation
- Supports keyword extraction with stopword removal
- Can summarize multiple files in a folder

## How to Run

```bash
git clone <repo-url>
cd AI-CHAT-SUMMARIZER
pip install -r requirements.txt
python main.py
