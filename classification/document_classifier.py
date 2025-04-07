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
import logging

class DocumentClassifier:
    def __init__(self, models_dir='models'):
        # Use same directory name format as in the training data
        self.categories = ['invoices', 'Shipping orders', 'resumes', 'others']
        self.models_dir = models_dir
        
        # Model paths
        self.invoice_model_path = os.path.join(models_dir, 'invoice_model.pkl')
        self.shipping_orders_model_path = os.path.join(models_dir, 'shipping_orders_model.pkl')
        self.resume_model_path = os.path.join(models_dir, 'resume_model.pkl')
        
        # Initialize models
        self.invoice_model = None
        self.invoice_vectorizer = None
        self.shipping_orders_model = None
        self.shipping_orders_vectorizer = None
        self.resume_model = None
        self.resume_vectorizer = None
        
        # Define key indicators for each category
        self.invoice_indicators = ['invoice', 'bill', 'receipt', 'payment', 'due date', 'invoice no', 'total amount', 'subtotal', 'tax']
        self.shipping_orders_indicators = ['shipping', 'order', 'shipment', 'delivery', 'tracking', 'package', 'carrier', 'freight', 'logistics', 'consignment']
        self.resume_indicators = ['resume', 'cv', 'curriculum vitae', 'experience', 'skills', 'education', 'employment', 'profile', 'objective', 'references']
        
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
                logging.info("Invoice model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading invoice model: {e}")
        
        # Load shipping orders model
        if os.path.exists(self.shipping_orders_model_path):
            try:
                with open(self.shipping_orders_model_path, 'rb') as f:
                    self.shipping_orders_vectorizer, self.shipping_orders_model = pickle.load(f)
                logging.info("Shipping orders model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading shipping orders model: {e}")
        
        # Load resume model
        if os.path.exists(self.resume_model_path):
            try:
                with open(self.resume_model_path, 'rb') as f:
                    self.resume_vectorizer, self.resume_model = pickle.load(f)
                logging.info("Resume model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading resume model: {e}")
    
    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stopwords, and lowercasing"""
        # Handle None or empty text
        if not text:
            return ""
            
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
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + " "
            
            # If PyPDF2 fails to extract text properly, try to log this
            if not text.strip():
                logging.warning(f"No text extracted from PDF {pdf_path}")
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
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
            logging.error(f"Error extracting text from image {image_path}: {e}")
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
            
            # Print full extracted text for debugging
            if text:
                logging.info(f"Extracted full text from {file_path}:\n{text}")
            else:
                logging.warning(f"No text extracted from {file_path}")
                
            return self.preprocess_text(text)
        except Exception as e:
            logging.error(f"Error extracting features from {file_path}: {e}")
            return ""
    
    def classify(self, file_path):
        """Classify a document into one of the categories"""
        # Extract features from the document
        features = self.extract_features(file_path)
        
        if not features:
            logging.warning(f"No features extracted from {file_path}")
            return "others"  # Default for unsupported files or empty content
        
        # Initialize scores for each category
        scores = {
            'invoices': 0.0,
            'Shipping orders': 0.0,  # Match the directory name
            'resumes': 0.0,
            'others': 0.25  # Higher default score for others to prevent misclassification
        }
        
        # Use invoice model if available
        if self.invoice_model is not None and self.invoice_vectorizer is not None:
            X = self.invoice_vectorizer.transform([features])
            invoice_prob = self.invoice_model.predict_proba(X)[0][1]  # Probability of being an invoice
            scores['invoices'] = invoice_prob
            logging.info(f"Invoice score: {invoice_prob:.4f}")
        
        # Use shipping orders model if available
        if self.shipping_orders_model is not None and self.shipping_orders_vectorizer is not None:
            X = self.shipping_orders_vectorizer.transform([features])
            shipping_orders_prob = self.shipping_orders_model.predict_proba(X)[0][1]  # Probability of being a shipping order
            scores['Shipping orders'] = shipping_orders_prob
            logging.info(f"Shipping orders score: {shipping_orders_prob:.4f}")
        
        # Use resume model if available
        if self.resume_model is not None and self.resume_vectorizer is not None:
            X = self.resume_vectorizer.transform([features])
            resume_prob = self.resume_model.predict_proba(X)[0][1]  # Probability of being a resume
            scores['resumes'] = resume_prob
            logging.info(f"Resume score: {resume_prob:.4f}")
        
        # No longer printing initial model scores
        
        # Keyword-based score adjustment
        original_text = ''
        _, ext = os.path.splitext(file_path)
        try:
            if ext.lower() == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page_text = pdf_reader.pages[page_num].extract_text()
                        if page_text:
                            original_text += page_text + " "
            elif ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                original_text = self.extract_text_from_image(file_path)
            elif ext.lower() in ['.txt', '.md', '.html']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    original_text = file.read()
        except Exception as e:
            logging.error(f"Error reading file for keyword analysis: {e}")
            
        # Apply keyword-based adjustments - boosting and penalizing
        if original_text:
            lower_text = original_text.lower()
            
            # Check for indicators presence
            has_invoice_indicators = any(keyword in lower_text for keyword in self.invoice_indicators)
            has_shipping_indicators = any(keyword in lower_text for keyword in self.shipping_orders_indicators)
            has_resume_indicators = any(keyword in lower_text for keyword in self.resume_indicators)
            
            # Penalize categories with no indicators but high scores
            if scores['invoices'] > 0.5 and not has_invoice_indicators:
                logging.info(f"High invoice score but no invoice indicators found - reducing score")
                scores['invoices'] *= 0.3  # Reduce the score by 70%
                scores['others'] += 0.3    # Boost others score
            
            if scores['Shipping orders'] > 0.5 and not has_shipping_indicators:
                logging.info(f"High shipping orders score but no shipping indicators found - reducing score")
                scores['Shipping orders'] *= 0.3
                scores['others'] += 0.3
                
            if scores['resumes'] > 0.5 and not has_resume_indicators:
                logging.info(f"High resume score but no resume indicators found - reducing score")
                scores['resumes'] *= 0.3
                scores['others'] += 0.3
            
            # Boost scores for categories with indicators
            invoice_boost = 0
            for keyword in self.invoice_indicators:
                if keyword in lower_text:
                    invoice_boost += 0.15  # Increase score for each keyword found
                    logging.info(f"Found invoice keyword: '{keyword}' - boosting invoice score")
            
            if invoice_boost > 0:
                scores['invoices'] = min(0.95, scores['invoices'] + invoice_boost)  # Cap at 0.95
            
            shipping_boost = 0
            for keyword in self.shipping_orders_indicators:
                if keyword in lower_text:
                    shipping_boost += 0.15  # Increase score for each keyword found
                    logging.info(f"Found shipping keyword: '{keyword}' - boosting shipping orders score")
            
            if shipping_boost > 0:
                scores['Shipping orders'] = min(0.95, scores['Shipping orders'] + shipping_boost)  # Cap at 0.95
                
            resume_boost = 0
            for keyword in self.resume_indicators:
                if keyword in lower_text:
                    resume_boost += 0.1  # Smaller boost for resume keywords
                    logging.info(f"Found resume keyword: '{keyword}' - boosting resume score")
            
            if resume_boost > 0:
                scores['resumes'] = min(0.9, scores['resumes'] + resume_boost)  # Cap at 0.9
        
        # Log adjusted scores
        logging.info(f"Adjusted scores after keyword analysis: {scores}")
        
        # If filename contains clear indicators, boost further
        filename = os.path.basename(file_path).lower()
        if 'invoice' in filename:
            scores['invoices'] = max(scores['invoices'], 0.8)  # Ensure high score but not override
            logging.info(f"Filename contains 'invoice' - ensuring high invoice score: {scores['invoices']}")
        elif 'shipping' in filename or 'order' in filename:
            scores['Shipping orders'] = max(scores['Shipping orders'], 0.8)
            logging.info(f"Filename contains shipping keywords - ensuring high shipping score: {scores['Shipping orders']}")
        elif 'resume' in filename or 'cv' in filename:
            scores['resumes'] = max(scores['resumes'], 0.8)
            logging.info(f"Filename contains resume keywords - ensuring high resume score: {scores['resumes']}")
        
        # If no category has high confidence, use rule-based classification
        highest_score = max(scores.values())
        if highest_score < 0.5:
            logging.info("No category has high confidence - using rule-based classification")
            rule_based_result = self._rule_based_classification(features, file_path)
            logging.info(f"Rule-based classification result: {rule_based_result}")
            return rule_based_result
        
        # Return the category with the highest probability
        result = max(scores, key=scores.get)
        logging.info(f"Final classification: {result} with score {scores[result]:.4f}")
        return result
    
    def _rule_based_classification(self, text, file_path):
        """Rule-based classification using keyword matching and patterns"""
        text = text.lower()
        filename = os.path.basename(file_path).lower()
        
        # Check for invoice indicators
        invoice_score = sum(1 for indicator in self.invoice_indicators if indicator in text)
        
        # Check for shipping order indicators
        shipping_orders_score = sum(1 for indicator in self.shipping_orders_indicators if indicator in text)
        
        # Check for resume document indicators
        resume_score = sum(1 for indicator in self.resume_indicators if indicator in text)
        
        # Calculate normalized scores
        total_invoice = len(self.invoice_indicators)
        total_shipping = len(self.shipping_orders_indicators)
        total_resume = len(self.resume_indicators)
        
        # Calculate the scores with higher threshold for classification
        scores = {
            'invoices': invoice_score / total_invoice if total_invoice > 0 else 0,
            'Shipping orders': shipping_orders_score / total_shipping if total_shipping > 0 else 0,
            'resumes': resume_score / total_resume if total_resume > 0 else 0,
            'others': 0.3  # Higher baseline for others
        }
        
        # Add filename-based boosts
        if 'invoice' in filename:
            scores['invoices'] += 0.2
        elif 'shipping' in filename or 'order' in filename:
            scores['Shipping orders'] += 0.2
        elif 'resume' in filename or 'cv' in filename:
            scores['resumes'] += 0.2
            
        # If no category has a good score, default to others
        max_score = max(scores['invoices'], scores['Shipping orders'], scores['resumes'])
        if max_score < 0.3:  # Low confidence in any specific category
            logging.info("No strong category indicators found in rule-based classification")
            return 'others'
            
        # Log the rule-based scores
        logging.info(f"Rule-based scores: {scores}")
        
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