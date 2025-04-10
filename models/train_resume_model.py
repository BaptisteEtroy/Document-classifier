import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import list_files_in_directory

def preprocess_text(text, stop_words):
    """Preprocess text by tokenizing, removing stopwords, and lowercasing"""
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
    import PyPDF2
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def extract_text_from_image(image_path):
    """Extract text from image using OCR"""
    import pytesseract
    from PIL import Image
    import cv2
    
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing to improve OCR accuracy
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # OCR
        text = pytesseract.image_to_string(gray)
        return text
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        return ""

def extract_features(file_path, stop_words):
    """Extract features from document based on file extension"""
    _, ext = os.path.splitext(file_path)
    
    try:
        if ext.lower() == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            text = extract_text_from_image(file_path)
        elif ext.lower() in ['.txt', '.md', '.html']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
        else:
            # For unsupported file types, return empty string
            return ""
        
        return preprocess_text(text, stop_words)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

def train_resume_model(data_dir='data', output_model_path='models/resume_model.pkl'):
    """Train a model to identify resume documents"""
    
    # Download NLTK resources if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    
    # Collect training data
    resume_dir = os.path.join(data_dir, 'resumes')
    
    other_files = []
    for category in ['invoices', 'Shipping orders', 'others']:
        category_dir = os.path.join(data_dir, category)
        if os.path.exists(category_dir):
            other_files.extend(list_files_in_directory(category_dir))
    
    if not os.path.exists(resume_dir):
        print(f"Resume directory {resume_dir} not found.")
        print(f"Please make sure either '{data_dir}/medical' or '{data_dir}/resumes' exists with resume documents.")
        return
    
    resume_files = list_files_in_directory(resume_dir)
    
    if not resume_files:
        print(f"No resume files found in {resume_dir}. Please add some and run again.")
        return
    
    if not other_files:
        print("No non-resume files found for negative examples. Using only resume features.")
    
    # Create dataset
    X = []
    y = []
    
    # Process resume files (positive examples)
    for file_path in resume_files:
        features = extract_features(file_path, stop_words)
        if features:
            X.append(features)
            y.append(1)  # 1 for resume
    
    # Process non-resume files (negative examples)
    for file_path in other_files:
        features = extract_features(file_path, stop_words)
        if features:
            X.append(features)
            y.append(0)  # 0 for non-resume
    
    if len(X) == 0:
        print("No valid documents found for training. Please check file formats.")
        return
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    except ValueError as e:
        print(f"Error splitting dataset: {e}")
        print("Training on all data without test split.")
        X_train, X_test, y_train, y_test = X_tfidf, X_tfidf, y, y
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Resume model accuracy: {accuracy:.4f}")
    
    # Save the model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    with open(output_model_path, 'wb') as f:
        pickle.dump((vectorizer, model), f)
    
    print(f"Resume model saved to {output_model_path}")

if __name__ == "__main__":
    train_resume_model() 