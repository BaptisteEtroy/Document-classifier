import os
import re
import json
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import numpy as np
import os.path
import pytesseract
from PIL import Image
import cv2

class DocumentClassifier:
    def __init__(self, models_dir='app/models'):
        self.categories = ['invoices', 'contracts', 'medical', 'others']
        self.models_dir = models_dir
        
        # Model paths
        self.invoice_model_path = os.path.join(models_dir, 'invoice_model.pkl')
        self.contract_model_path = os.path.join(models_dir, 'contract_model.pkl')
        self.medical_model_path = os.path.join(models_dir, 'medical_model.pkl')
        
        # Initialize models
        self.invoice_model = None
        self.invoice_vectorizer = None
        self.contract_model = None
        self.contract_vectorizer = None
        self.medical_model = None
        self.medical_vectorizer = None
        
        # Download NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        # Load models if they exist
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        # Load invoice model
        if os.path.exists(self.invoice_model_path):
            try:
                with open(self.invoice_model_path, 'rb') as f:
                    self.invoice_vectorizer, self.invoice_model = pickle.load(f)
                print("Invoice model loaded successfully")
            except Exception as e:
                print(f"Error loading invoice model: {e}")
        
        # Load contract model
        if os.path.exists(self.contract_model_path):
            try:
                with open(self.contract_model_path, 'rb') as f:
                    self.contract_vectorizer, self.contract_model = pickle.load(f)
                print("Contract model loaded successfully")
            except Exception as e:
                print(f"Error loading contract model: {e}")
        
        # Load medical model
        if os.path.exists(self.medical_model_path):
            try:
                with open(self.medical_model_path, 'rb') as f:
                    self.medical_vectorizer, self.medical_model = pickle.load(f)
                print("Medical model loaded successfully")
            except Exception as e:
                print(f"Error loading medical model: {e}")
    
    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stopwords, and lowercasing"""
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(tokens)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
        return text
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
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
    
    def extract_features(self, file_path):
        """Extract features from document based on file extension"""
        _, ext = os.path.splitext(file_path)
        
        try:
            if ext.lower() == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text = self.extract_text_from_image(file_path)
            elif ext.lower() in ['.txt', '.md', '.html']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
            else:
                # For unsupported file types, return empty string
                return ""
            
            return self.preprocess_text(text)
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return ""
    
    def classify(self, file_path):
        """Classify a document into one of the categories"""
        # Extract features from the document
        features = self.extract_features(file_path)
        
        if not features:
            return "others"  # Default for unsupported files or empty content
        
        # Initialize scores for each category
        scores = {
            'invoices': 0.0,
            'contracts': 0.0,
            'medical': 0.0,
            'others': 0.1  # Default score for others
        }
        
        # Use invoice model if available
        if self.invoice_model is not None and self.invoice_vectorizer is not None:
            X = self.invoice_vectorizer.transform([features])
            invoice_prob = self.invoice_model.predict_proba(X)[0][1]  # Probability of being an invoice
            scores['invoices'] = invoice_prob
        
        # Use contract model if available
        if self.contract_model is not None and self.contract_vectorizer is not None:
            X = self.contract_vectorizer.transform([features])
            contract_prob = self.contract_model.predict_proba(X)[0][1]  # Probability of being a contract
            scores['contracts'] = contract_prob
        
        # Use medical model if available
        if self.medical_model is not None and self.medical_vectorizer is not None:
            X = self.medical_vectorizer.transform([features])
            medical_prob = self.medical_model.predict_proba(X)[0][1]  # Probability of being a medical document
            scores['medical'] = medical_prob
        
        # If no model provided high confidence, use rule-based classification
        max_prob = max(scores.values())
        if max_prob < 0.5:
            return self._rule_based_classification(features, file_path)
        
        # Return the category with the highest probability
        return max(scores, key=scores.get)
    
    def _rule_based_classification(self, text, file_path):
        """Rule-based classification using keyword matching and patterns"""
        text = text.lower()
        
        # Check for invoice indicators
        invoice_indicators = ['invoice', 'bill', 'receipt', 'payment', 'due date', 'invoice no', 'total amount', 'subtotal', 'tax']
        invoice_score = sum(1 for indicator in invoice_indicators if indicator in text)
        
        # Check for contract indicators
        contract_indicators = ['contract', 'agreement', 'terms', 'conditions', 'parties', 'hereby', 'provisions', 'clause', 'shall', 'legal']
        contract_score = sum(1 for indicator in contract_indicators if indicator in text)
        
        # Check for medical document indicators
        medical_indicators = ['patient', 'diagnosis', 'treatment', 'medical', 'doctor', 'hospital', 'clinic', 'prescription', 'symptoms', 'health']
        medical_score = sum(1 for indicator in medical_indicators if indicator in text)
        
        # Calculate the scores
        scores = {
            'invoices': invoice_score / len(invoice_indicators) if invoice_indicators else 0,
            'contracts': contract_score / len(contract_indicators) if contract_indicators else 0,
            'medical': medical_score / len(medical_indicators) if medical_indicators else 0,
            'others': 0.1  # Default score for others
        }
        
        # Classify based on the highest score
        return max(scores, key=scores.get)
    
    def save_result(self, file_path, category, output_dir):
        """Save the classification result to a JSON file"""
        base_filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"{base_filename}_classification.json")
        
        result = {
            'file': file_path,
            'category': category,
            'timestamp': os.path.getmtime(file_path)
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        return output_path 