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

def train_shipping_orders_model(data_dir='data', output_model_path='models/shipping_orders_model.pkl'):
    """Train a model to identify shipping orders documents"""
    
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
    shipping_orders_dir = os.path.join(data_dir, 'Shipping orders')
    other_files = []
    for category in ['invoices', 'resumes', 'others']:
        category_dir = os.path.join(data_dir, category)
        if os.path.exists(category_dir):
            other_files.extend(list_files_in_directory(category_dir))
    
    if not os.path.exists(shipping_orders_dir):
        print(f"Shipping orders directory {shipping_orders_dir} not found. Creating it...")
        os.makedirs(shipping_orders_dir, exist_ok=True)
        print(f"Please add shipping order documents to {shipping_orders_dir} and run again.")
        return
    
    shipping_orders_files = list_files_in_directory(shipping_orders_dir)
    
    if not shipping_orders_files:
        print(f"No shipping order files found in {shipping_orders_dir}. Please add some and run again.")
        return
    
    if not other_files:
        print("No non-shipping-order files found for negative examples. Using only shipping order features.")
    
    # Create dataset
    X = []
    y = []
    
    # Process shipping order files (positive examples)
    for file_path in shipping_orders_files:
        features = extract_features(file_path, stop_words)
        if features:
            X.append(features)
            y.append(1)  # 1 for shipping order
    
    # Process non-shipping-order files (negative examples)
    for file_path in other_files:
        features = extract_features(file_path, stop_words)
        if features:
            X.append(features)
            y.append(0)  # 0 for non-shipping-order
    
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
    print(f"Shipping orders model accuracy: {accuracy:.4f}")
    
    # Save the model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    with open(output_model_path, 'wb') as f:
        pickle.dump((vectorizer, model), f)
    
    print(f"Shipping orders model saved to {output_model_path}")

if __name__ == "__main__":
    train_shipping_orders_model() 