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
        # Regex patterns for extraction - simplified for better accuracy
        self.patterns = {
            'invoice_number': [
                r'(?:invoice|inv)(?:oice)?\s*(?:number|no|#)?\s*:?\s*(\d+)',
                r'(?:invoice|inv)(?:oice)?\s*(?:number|no|#)?\s*:?\s*([A-Z0-9][\w\d\-\/\.]+)',
                r'(?:number|no|#)\s*:?\s*(\d+)',
                r'(?:invoice|inv)(?:oice)?\s*(?:number|no|#)?[^\n]*?(\d{5,})'
            ],
            'invoice_date': [
                r'(?:date\s+of\s+issue|invoice\s+date|date)[:;]\s*(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'(?:date\s+of\s+issue|invoice\s+date|date)[^\n]*?(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'(?:date\s+of\s+issue|invoice\s+date|date)[:;]?\s*(\w+\s+\d{1,2},?\s+\d{4})'
            ],
            'due_date': [
                r'(?:due\s+date|payment\s+date)[:;]\s*(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'(?:due\s+date|payment\s+date)[^\n]*?(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'(?:due\s+date|due|payment\s+due)[:;]?\s*(\w+\s+\d{1,2},?\s+\d{4})',
                r'(?:due|payment)[^\n]*?(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})'
            ],
            'issuer_name': [
                r'^([A-Z][A-Za-z0-9\s]{2,}(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)\s*$',
                r'(?:from|company)[:;]?\s*([A-Z][A-Za-z0-9\s]{2,}(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)'
            ],
            'recipient_name': [
                r'(?:bill(?:ed)?\s+to|to)[:;]?\s*([A-Z][A-Za-z0-9\s]{2,})',
                r'(?:bill(?:ed)?\s+to|to)[:;]?\s*\n\s*([A-Za-z0-9\s]{2,})',
                r'(?:recipient|client|customer)[:;]?\s*([A-Za-z0-9\s]{2,})'
            ],
            'total_amount': [
                r'(?:invoice\s+total|total|amount\s+due)[:;]?\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'(?:invoice\s+total|total|amount\s+due)[^\n]*?\$?\s*([\d,]+(?:\.\d{2})?)',
                r'\$\s*([\d,]+(?:\.\d{2})?)(?:\s*(?:usd|eur|gbp|cad))?$',
                r'(?:^|\n)\s*\$\s*([\d,]+(?:\.\d{2})?)(?:\s*(?:usd|eur|gbp|cad))?(?:$|\n)',
                r'(?:^|\n)[^\n]*?total[^\n]*?\$\s*([\d,]+(?:\.\d{2})?)(?:\s*(?:usd|eur|gbp|cad))?'
            ],
            'subtotal': [
                r'(?:sub[-\s]?total)[:;]?\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'(?:sub[-\s]?total)[^\n]*?\$?\s*([\d,]+(?:\.\d{2})?)'
            ],
            'tax': [
                r'(?:tax|vat|gst|sales\s*tax)(?:\s*\(\d+%\))?[:;]?\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'(?:tax|vat|gst|sales\s*tax)[^\n]*?\$?\s*([\d,]+(?:\.\d{2})?)'
            ],
            'email': [
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'(?:e[-]?mail|contact)[:;]?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ],
            'phone_number': [
                r'(?:phone|tel|telephone|contact)[:;]?\s*(\+?\d{1,3}[-\.\s]?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4})',
                r'(?:phone|tel|telephone|contact)[^\n]*?(\+?\d{1,3}[-\.\s]?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4})',
                r'(\d{3}[-\.\s]\d{3}[-\.\s]\d{4})'
            ],
            'payment_terms': [
                r'(?:terms|payment\s+terms|net)[:;]?\s*([A-Za-z0-9\s]{1,30})',
                r'(?:net|payment\s+terms)[^\n]*?(\d{1,3}(?:\s*days)?)'
            ]
        }
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                print(f"Warning: No text extracted from PDF {pdf_path}")
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
        return text
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR with improved preprocessing"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path}")
                return ""
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing to improve OCR accuracy
            # 1. Standard binary threshold with Otsu's method
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Create a config for better OCR results
            custom_config = r'--oem 3 --psm 6'
            
            # OCR with tesseract
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            # Keep original line breaks for better text structure recognition
            return text
        except Exception as e:
            print(f"Error extracting text from image {image_path}: {e}")
            return ""
    
    def extract_field(self, text, field_name):
        """Extract a specific field using regex patterns"""
        for pattern in self.patterns[field_name]:
            matches = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                value = matches.group(1).strip()
                
                # Clean up the extracted value
                if field_name in ['invoice_date', 'due_date']:
                    try:
                        parsed_date = dateutil.parser.parse(value)
                        value = parsed_date.strftime('%Y-%m-%d')
                    except:
                        # If parsing fails, keep the original format but clean it
                        value = re.sub(r'[^\d\/\-\.]', '', value)
                elif field_name in ['total_amount', 'subtotal', 'tax']:
                    # Clean amount values (remove commas, ensure decimal)
                    value = value.replace(',', '')
                    # Make sure we have a decimal point for money values
                    if '.' not in value and value.strip():
                        value = f"{value}.00"  # Add decimal if missing
                
                return value
        return None
    
    def extract_large_dollar_amount(self, text):
        """Find large dollar amounts (likely to be the total)"""
        # Look for dollar amounts that are likely to be the total
        # Prioritize amounts with dollar signs, especially if they're isolated
        amount_matches = re.findall(r'\$\s*([\d,]+(?:\.\d{2})?)', text)
        if amount_matches:
            # Try to find the largest amount
            try:
                clean_amounts = [float(amt.replace(',', '')) for amt in amount_matches]
                if clean_amounts:
                    # Prefer amounts that are multiples of 100 or 1000 (more likely to be totals)
                    large_amounts = [amt for amt in clean_amounts if amt >= 100]
                    if large_amounts:
                        return str(max(large_amounts))
                    return str(max(clean_amounts))
            except:
                pass
        
        # Look for isolated numbers in the last part of the document
        last_part = text[int(len(text)*0.7):]  # Check the last 30% of the text
        # Look for numbers that might be totals (isolated on lines, larger values)
        isolated_amounts = re.findall(r'(?:^|\n)\s*\$?\s*([\d,]+(?:\.\d{2})?)\s*(?:$|\n)', last_part)
        if isolated_amounts:
            try:
                clean_amounts = [float(amt.replace(',', '')) for amt in isolated_amounts]
                if clean_amounts:
                    return str(max(clean_amounts))
            except:
                pass
        
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
        
        # Check if we got any text to process
        if not text or len(text.strip()) < 10:
            return {
                'file': file_path,
                'error': 'No text could be extracted or text is too short',
                'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Print the first 200 chars for debugging
        print(f"Extracted Text (first 200 chars): {text[:200]}")
        
        # Extract each field
        results = {
            'file': file_path,
            'invoice_number': self.extract_field(text, 'invoice_number'),
            'invoice_date': self.extract_field(text, 'invoice_date'),
            'due_date': self.extract_field(text, 'due_date'),
            'issuer_name': self.extract_field(text, 'issuer_name'),
            'recipient_name': self.extract_field(text, 'recipient_name'),
            'total_amount': self.extract_field(text, 'total_amount'),
            'subtotal': self.extract_field(text, 'subtotal'),
            'tax': self.extract_field(text, 'tax'),
            'payment_terms': self.extract_field(text, 'payment_terms'),
            'email': self.extract_field(text, 'email'),
            'phone_number': self.extract_field(text, 'phone_number'),
            'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Cleanup and post-processing
        
        # Look for dollar signs followed by numbers for total amount if not found
        if not results['total_amount']:
            large_amount = self.extract_large_dollar_amount(text)
            if large_amount:
                results['total_amount'] = large_amount
                
        # Look for clear invoice number patterns
        if not results['invoice_number']:
            # Check for a standalone number after "Invoice Number" or variations
            inv_num_match = re.search(r'(?:invoice|inv)(?:oice)?\s*(?:number|no|#)?[^\n]*?(\d{4,})', text, re.IGNORECASE)
            if inv_num_match:
                results['invoice_number'] = inv_num_match.group(1)
            # Try looking for numbers at the top of the doc
            elif "invoice" in text.lower():
                # Find the line with invoice and look for nearby numbers
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if "invoice" in line.lower():
                        # Check nearby lines for numbers
                        for j in range(max(0, i-2), min(len(lines), i+3)):
                            num_match = re.search(r'(\d{4,})', lines[j])
                            if num_match:
                                results['invoice_number'] = num_match.group(1)
                                break
                        if results['invoice_number']:
                            break
        
        # Look for company name at the top
        if not results['issuer_name']:
            lines = text.split('\n')
            first_lines = [line for line in lines[:5] if len(line.strip()) > 3]
            if first_lines:
                results['issuer_name'] = first_lines[0].strip()
                
        # Get recipient name from "Bill To" section
        if not results['recipient_name']:
            bill_to_match = re.search(r'(?:bill(?:ed)?\s+to|to)[^\n]*?\n\s*([A-Za-z0-9\s]{2,})', text, re.IGNORECASE)
            if bill_to_match:
                results['recipient_name'] = bill_to_match.group(1).strip()
        
        # Look for email specifically
        if not results['email']:
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
            if email_match:
                results['email'] = email_match.group(0)
        
        # Clean up None values for JSON serialization and trim long values
        for key, value in results.items():
            if value is None:
                results[key] = ""
            elif isinstance(value, str) and len(value) > 100:
                # Truncate overly long values and keep just the first line
                first_line = value.split('\n')[0]
                results[key] = first_line[:100]
                
        return results
    
    def save_results(self, results, output_path):
        """Save the extraction results to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return output_path