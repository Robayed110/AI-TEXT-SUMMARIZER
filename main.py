import re

import os
import nltk
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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


# -------- Message Statistics --------
def message_stats(user_msgs, ai_msgs):
    return {
        "total_messages": len(user_msgs) + len(ai_msgs),
        "user_messages": len(user_msgs),
        "ai_messages": len(ai_msgs)
    }


# -------- Simple Keyword Extraction --------
def extract_keywords(messages, top_n=5):
    text = " ".join(messages).lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w.isalnum() and w not in stop_words]
    freq = Counter(filtered_words)
    return freq.most_common(top_n)

# -------- TF-IDF Keyword Extraction (Optional) --------
def tfidf_keywords(messages, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(messages)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    word_scores = list(zip(feature_names, scores))
    sorted_keywords = sorted(word_scores, key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]


# -------- Generate and Print Summary --------
def generate_summary(file_name, stats, keywords):
    total = stats["total_messages"]
    topic = ", ".join([kw for kw, _ in keywords[:2]]) if keywords else "general topics"
    keyword_list = ", ".join([kw for kw, _ in keywords]) if keywords else "N/A"

    summary = [
        "Summary:",
        f"- The conversation had {total} exchanges.",
        f"- The user asked mainly about {topic}.",
        f"- Most common keywords: {keyword_list}."
    ]

    print("\n".join(summary))

    # Save to output file
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"{os.path.splitext(file_name)[0]}_summary.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary))



# -------- Process a Single File --------
def process_file(file_path, use_tfidf=False):
    user_msgs, ai_msgs = parse_chat_log(file_path)
    stats = message_stats(user_msgs, ai_msgs)
    messages = user_msgs + ai_msgs

    keywords = tfidf_keywords(messages) if use_tfidf else extract_keywords(messages)
    generate_summary(os.path.basename(file_path), stats, keywords)




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
