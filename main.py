import re

import os
from collections import defaultdict

def parse_chat_log(file_path):
    user_msgs = []
    ai_msgs = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("User:"):
                user_msgs.append(line[5:].strip())
            elif line.startswith("AI:"):
                ai_msgs.append(line[3:].strip())
    return user_msgs, ai_msgs


# -------- Process a Single File --------
def process_file(file_path, use_tfidf=False):
    user_msgs, ai_msgs = parse_chat_log(file_path)
    print(user_msgs, ai_msgs)



# -------- Process All Chat Logs in a Folder --------
def process_all_logs(folder_path='chat_logs', use_tfidf=False):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found.")
        return

    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"No .txt files found in '{folder_path}'.")
        return

    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"\nProcessing: {file_name}")
        process_file(file_path, use_tfidf)



# -------- Entry Point --------
if __name__ == "__main__":
    print("AI Chat Log Summarizer\n")

    # If use TF-IDF make it true
    USE_TFIDF = False

    # Process all logs in the folder
    process_all_logs(folder_path='chat_logs', use_tfidf=USE_TFIDF)
