import os
import re
import json
import PyPDF2
import dateutil.parser
from datetime import datetime
import pytesseract
from PIL import Image
import cv2
import numpy as np

class InvoiceExtractor:
    def __init__(self):
        # Regex patterns for extraction
        self.patterns = {
            'invoice_number': [
                r'(?:invoice|inv)(?:oice)?(?:\s+|-|#|number|num|no)(?:\s|-|:|\.|#)*\s*([A-Z0-9][\w\d\-\/\.]+)',
                r'(?:invoice|inv)(?:oice)?\s*(?:number|num|no)?(?:\s|-|:|\.|#)?\s*([A-Z0-9][\w\d\-\/\.]+)',
                r'invoice\s*#\s*(\w+)'
            ],
            'invoice_date': [
                r'(?:invoice|inv)(?:oice)?(?:\s+|-|#|number|num|no)?(?:\s+|-|:|\.)?\s*date(?:\s+|-|:|\.)?\s*(\d{1,4}[-\.\/]\d{1,2}[-\.\/]\d{1,4})',
                r'date(?:\s+|-|:|\.)?\s*(\d{1,4}[-\.\/]\d{1,2}[-\.\/]\d{1,4})',
                r'date(?:\s+|-|:|\.)?\s*(\w+\s+\d{1,2},?\s+\d{4})',
                r'date(?:\s+|-|:|\.)?\s*(\d{1,2}\s+\w+\s+\d{4})'
            ],
            'due_date': [
                r'(?:due|payment)(?:\s+|-|#|number|num|no)?(?:\s+|-|:|\.)?\s*date(?:\s+|-|:|\.)?\s*(\d{1,4}[-\.\/]\d{1,2}[-\.\/]\d{1,4})',
                r'due(?:\s+|-|:|\.)?\s*(?:date)?(?:\s+|-|:|\.)?\s*(\d{1,4}[-\.\/]\d{1,2}[-\.\/]\d{1,4})',
                r'due(?:\s+|-|:|\.)?\s*(?:date)?(?:\s+|-|:|\.)?\s*(\w+\s+\d{1,2},?\s+\d{4})',
                r'due(?:\s+|-|:|\.)?\s*(?:date)?(?:\s+|-|:|\.)?\s*(\d{1,2}\s+\w+\s+\d{4})'
            ],
            'issuer_name': [
                r'(?:from|sender|company|issuer|issued by|bill from|seller):?\s*([A-Z][A-Za-z0-9\s\.,&]+(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)',
                r'(?:^|\n)([A-Z][A-Za-z0-9\s\.,&]+(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)\s*\n'
            ],
            'recipient_name': [
                r'(?:to|bill to|recipient|customer|client|buyer):?\s*([A-Z][A-Za-z0-9\s\.,&]+(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)',
                r'(?:BILL TO|TO)(?:\s+|-|:|\.)?\s*([A-Z][A-Za-z0-9\s\.,&]+(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)'
            ],
            'total_amount': [
                r'(?:total|amount|sum|balance|due|payment)(?:\s+|-|:|\.)?\s*(?:due|amount|paid)?(?:\s+|-|:|\.)?\s*\$?\s*([\d,]+\.\d{2})',
                r'(?:total|amount|sum|balance|due|payment)(?:\s+|-|:|\.)?\s*(?:due|amount|paid)?(?:\s+|-|:|\.)?\s*\$?\s*([\d,]+)',
                r'\$\s*([\d,]+\.\d{2})',
                r'(?:TOTAL|Total)(?:\s+|-|:|\.)?\s*\$?\s*([\d,]+\.\d{2})'
            ],
        }
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing to improve OCR accuracy
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # OCR
        text = pytesseract.image_to_string(gray)
        return text
    
    def extract_field(self, text, field_name):
        """Extract a specific field using regex patterns"""
        for pattern in self.patterns[field_name]:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                value = matches.group(1).strip()
                
                # Clean up the extracted value
                if field_name in ['invoice_date', 'due_date']:
                    try:
                        parsed_date = dateutil.parser.parse(value)
                        value = parsed_date.strftime('%Y-%m-%d')
                    except:
                        pass
                elif field_name == 'total_amount':
                    value = value.replace(',', '')
                
                return value
        return None
    
    def extract(self, file_path):
        """Extract information from an invoice document"""
        _, ext = os.path.splitext(file_path)
        
        # Extract text based on file type
        if ext.lower() == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            text = self.extract_text_from_image(file_path)
        elif ext.lower() in ['.txt', '.md', '.html']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
        else:
            return {'error': f'Unsupported file type: {ext}'}
        
        # Extract each field
        results = {
            'file': file_path,
            'invoice_number': self.extract_field(text, 'invoice_number'),
            'invoice_date': self.extract_field(text, 'invoice_date'),
            'due_date': self.extract_field(text, 'due_date'),
            'issuer_name': self.extract_field(text, 'issuer_name'),
            'recipient_name': self.extract_field(text, 'recipient_name'),
            'total_amount': self.extract_field(text, 'total_amount'),
            'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return results
    
    def save_results(self, results, output_path):
        """Save the extraction results to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return output_path 