import re
import nltk
from nltk.corpus import stopwords

# Download stopwords first time only
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Lowercase, remove non-alphabetic characters, and remove stopwords.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove numbers/punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)