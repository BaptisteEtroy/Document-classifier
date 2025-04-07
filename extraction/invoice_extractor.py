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
                # Very specific invoice number patterns first
                r'invoice\s+no\.?\s*(\d+)',
                r'inv\s+no\.?\s*(\d+)',
                r'invoice\s+number\s*:?\s*(\d+)',
                r'invoice\s+#\s*(\d+)',
                r'invoice\s*:\s*(\d+)',
                r'inv\s*#\s*(\d+)',
                # More general patterns
                r'no\.?\s*(\d+)',
                r'number\s*:?\s*(\d+)',
                r'#\s*(\d+)',
                # Alphanumeric invoice numbers
                r'invoice\s+no\.?\s*([A-Z0-9][\w\d\-\/\.]+)',
                r'inv\s+no\.?\s*([A-Z0-9][\w\d\-\/\.]+)',
                r'invoice\s+number\s*:?\s*([A-Z0-9][\w\d\-\/\.]+)',
                r'invoice\s+#\s*([A-Z0-9][\w\d\-\/\.]+)',
                r'no\.?\s*([A-Z0-9][\w\d\-\/\.]+)',
                # Very generic patterns as last resort
                r'(?:invoice|inv)[^\d\n]{1,20}(\d{4,})'
            ],
            'invoice_date': [
                r'invoice\s+date\s*:?\s*(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'date\s+of\s+invoice\s*:?\s*(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'date\s*:?\s*(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'invoice\s+date\s*:?\s*(\d{1,2}\s+\w+\s+\d{4})',
                r'invoice\s+date\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})',
                r'date\s*:?\s*(\d{1,2}\s+\w+\s+\d{4})',
                r'date\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})'
            ],
            'due_date': [
                r'due\s+date\s*:?\s*(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'payment\s+due\s*:?\s*(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'due\s+on\s*:?\s*(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
                r'due\s+date\s*:?\s*(\d{1,2}\s+\w+\s+\d{4})',
                r'payment\s+due\s*:?\s*(\d{1,2}\s+\w+\s+\d{4})',
                r'due\s+on\s*:?\s*(\d{1,2}\s+\w+\s+\d{4})',
                r'due\s+date\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})',
                r'payment\s+due\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})',
                r'due\s+on\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})'
            ],
            'issuer_name': [
                r'from\s*:?\s*([A-Z][A-Za-z0-9\s,\.]{2,}(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)',
                r'seller\s*:?\s*([A-Z][A-Za-z0-9\s,\.]{2,}(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)',
                r'vendor\s*:?\s*([A-Z][A-Za-z0-9\s,\.]{2,}(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)',
                r'billed\s+from\s*:?\s*([A-Z][A-Za-z0-9\s,\.]{2,}(?:LLC|Inc|Ltd|Corp|Corporation|Company|Co)?)'
            ],
            'recipient_name': [
                r'(?:bill(?:ed)?\s+to|to|customer|client|buyer)\s*:?\s*([A-Z][A-Za-z0-9\s,\.]{2,})',
                r'(?:bill(?:ed)?\s+to|to|customer|client|buyer)\s*:?\s*\n\s*([A-Za-z0-9\s,\.]{2,})',
                r'(?:recipient|attention|attn)\s*:?\s*([A-Za-z0-9\s,\.]{2,})'
            ],
            'total_amount': [
                r'(?:total|amount\s+due|balance\s+due|total\s+due)\s*:?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)',
                r'(?:total|amount\s+due|balance\s+due|total\s+due)[^\n]*?\$?\s*([\d,]+(?:\.\d{1,2})?)',
                r'\$\s*([\d,]+(?:\.\d{1,2})?)(?:\s*(?:usd|eur|gbp|cad))?(?:\s*$|\s*\n)',
                r'(?:usd|eur|gbp|cad)?\s*([\d,]+(?:\.\d{1,2})?)(?:\s*(?:usd|eur|gbp|cad))?(?:\s*$|\s*\n)'
            ],
            'subtotal': [
                r'(?:sub[-\s]?total|amount|net\s+amount)\s*:?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)',
                r'(?:sub[-\s]?total|amount|net\s+amount)[^\n]*?\$?\s*([\d,]+(?:\.\d{1,2})?)'
            ],
            'tax': [
                r'(?:tax|vat|gst|sales\s*tax|hst)(?:\s*\(\d+%\))?\s*:?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)',
                r'(?:tax|vat|gst|sales\s*tax|hst)[^\n]*?\$?\s*([\d,]+(?:\.\d{1,2})?)'
            ],
            'email': [
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'(?:e[-]?mail|contact|email\s+us|support)\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ],
            'phone_number': [
                r'(?:phone|tel|telephone|contact|call)\s*:?\s*(\+?\d{1,3}[-\.\s]?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4})',
                r'(?:phone|tel|telephone|contact|call)[^\n]*?(\+?\d{1,3}[-\.\s]?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4})',
                r'(\d{3}[-\.\s]\d{3}[-\.\s]\d{4})'
            ],
            'payment_terms': [
                r'(?:terms|payment\s+terms|net)\s*:?\s*([A-Za-z0-9\s]{1,30})',
                r'(?:net|payment\s+terms)[^\n]*?(\d{1,3}(?:\s*days)?)'
            ]
        }
        
        # Common header terms to filter out
        self.header_terms = [
            'invoice', 'bill', 'receipt', 'statement', 'billed to', 'bill to', 
            'ship to', 'payment details', 'from', 'to', 'payment info'
        ]
    
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
            # Try multiple preprocessing approaches for better results
            
            # 1. Standard binary threshold with Otsu's method
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # 2. Add optional image enhancement (adjust contrast)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Create configs for better OCR results
            configs = [
                r'--oem 3 --psm 6',  # Assume a single uniform block of text
                r'--oem 3 --psm 4'   # Assume a single column of text
            ]
            
            # Try different preprocessing methods and configs
            results = []
            for img, config in [(thresh, configs[0]), (enhanced_thresh, configs[0]), (enhanced_thresh, configs[1])]:
                result = pytesseract.image_to_string(img, config=config)
                results.append(result)
            
            # Use the result with the most text (likely the best extraction)
            text = max(results, key=len)
            
            # Print extracted text for debugging
            print(f"Extracted text from image:\n{text[:500]}...")
            
            return text
        except Exception as e:
            print(f"Error extracting text from image {image_path}: {e}")
            return ""
    
    def preprocess_text(self, text):
        """Apply preprocessing to improve text quality for extraction"""
        if not text:
            return ""
            
        # Replace common OCR errors
        text = re.sub(r'[$S]', '$', text)  # Common OCR error with S instead of $
        
        # Add spacing around key patterns to improve detection
        text = re.sub(r'(Invoice|No\.|Number|Date|Total|Amount|Due|Bill|To)', r' \1 ', text, flags=re.IGNORECASE)
        
        return text
    
    def is_header_text(self, text):
        """Check if text appears to be a header rather than actual data"""
        text = text.lower().strip()
        
        # Check if the text consists primarily of header keywords
        for term in self.header_terms:
            if term.lower() in text:
                # If the text is short and dominated by the header term, it's likely a header
                if len(text) < len(term) + 10:
                    return True
                    
        # Check for all caps with no digits (likely a header)
        if text.isupper() and not any(char.isdigit() for char in text):
            return True
            
        return False
    
    def extract_field(self, text, field_name):
        """Extract a specific field using regex patterns"""
        for pattern in self.patterns[field_name]:
            matches = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                value = matches.group(1).strip()
                
                # Clean up the extracted value
                if field_name in ['invoice_date', 'due_date']:
                    try:
                        parsed_date = dateutil.parser.parse(value, fuzzy=True)
                        value = parsed_date.strftime('%Y-%m-%d')
                    except:
                        # If parsing fails, keep the original format but clean it
                        value = re.sub(r'[^\d\/\-\.\s\w,]', '', value).strip()
                elif field_name in ['total_amount', 'subtotal', 'tax']:
                    # Clean amount values (remove commas, ensure decimal)
                    value = value.replace(',', '')
                    # Make sure we have a decimal point for money values
                    if '.' not in value and value.strip():
                        value = f"{value}.00"  # Add decimal if missing
                elif field_name in ['issuer_name', 'recipient_name']:
                    # Don't return header text as actual names
                    if self.is_header_text(value):
                        continue
                
                return value
        return ""
    
    def extract_invoice_number(self, text):
        """Extract invoice number with special handling for common formats"""
        # First try specific "Invoice No." pattern
        inv_no_match = re.search(r'invoice\s+no\.?\s*(\d+)', text, re.IGNORECASE)
        if inv_no_match:
            return inv_no_match.group(1).strip()
            
        # Then try just "No." followed by numbers
        no_match = re.search(r'no\.?\s*(\d+)', text, re.IGNORECASE)
        if no_match:
            return no_match.group(1).strip()
            
        # Then try standard patterns from self.patterns
        for pattern in self.patterns['invoice_number']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        # Last resort - look for "invoice" near a series of digits
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'invoice' in line.lower():
                # Check this line and nearby lines for numbers
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    num_match = re.search(r'(\d{4,})', lines[j])
                    if num_match:
                        return num_match.group(1).strip()
        
        return ""
    
    def extract_person_name(self, text):
        """Extract a person's name from text, assuming names follow standard capitalization"""
        # Look for patterns that might represent a person's name
        # Typically 2-3 capitalized words not preceded by common prefixes
        
        # First find potential names (2-3 capitalized words)
        potential_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', text)
        
        if potential_names:
            # Filter out names that appear immediately after common header words
            filtered_names = []
            header_prefixes = ['invoice', 'bill', 'to', 'from', 'date', 'due', 'attention', 'attn', 'customer', 'client']
            
            for name in potential_names:
                # Check if this name appears after a header prefix
                is_after_prefix = False
                for prefix in header_prefixes:
                    if re.search(r'\b' + re.escape(prefix) + r'\s+' + re.escape(name) + r'\b', text, re.IGNORECASE):
                        is_after_prefix = True
                        break
                
                if not is_after_prefix and not self.is_header_text(name) and len(name.split()) >= 2:
                    filtered_names.append(name)
            
            if filtered_names:
                return filtered_names[0].strip()
        
        return ""
    
    def extract_dates(self, text):
        """Extract all dates from the text and return them in order of appearance"""
        date_patterns = [
            r'(\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4})',
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{4})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,|st|nd|rd|th)?\s+\d{4})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2}(?:,|st|nd|rd|th)?\s+\d{4})'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                try:
                    parsed_date = dateutil.parser.parse(date_str, fuzzy=True)
                    formatted_date = parsed_date.strftime('%Y-%m-%d')
                    dates.append((formatted_date, match.start()))
                except:
                    # If parsing fails, skip this match
                    continue
        
        # Sort dates by position in text
        dates.sort(key=lambda x: x[1])
        
        # Return just the date strings
        return [date for date, _ in dates]
    
    def extract_amounts(self, text):
        """Extract all potential monetary amounts from text"""
        amount_patterns = [
            r'\$\s*([\d,]+(?:\.\d{1,2})?)',
            r'([\d,]+(?:\.\d{1,2})?)(?:\s*(?:USD|EUR|GBP|CAD|dollars|€|£))'
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    amounts.append((amount, match.start()))
                except:
                    continue
        
        # Sort amounts by value (largest first)
        amounts.sort(key=lambda x: x[0], reverse=True)
        
        # Return just the amount values as strings
        return [str(amount) for amount, _ in amounts]
    
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
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Initialize results dictionary
        results = {
            'file': file_path,
            'invoice_number': '',
            'invoice_date': '',
            'due_date': '',
            'issuer_name': '',
            'recipient_name': '',
            'total_amount': '',
            'subtotal': '',
            'tax': '',
            'payment_terms': '',
            'email': '',
            'phone_number': '',
            'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Extract invoice number - using specialized function
        results['invoice_number'] = self.extract_invoice_number(text)
        
        # Extract dates
        dates = self.extract_dates(text)
        
        # Assign dates based on context or position
        if dates:
            # First try to find specific invoice date and due date
            invoice_date = self.extract_field(text, 'invoice_date')
            due_date = self.extract_field(text, 'due_date')
            
            if invoice_date:
                results['invoice_date'] = invoice_date
            elif dates:
                # If no specific invoice date found, use first date in document
                results['invoice_date'] = dates[0]
                
            if due_date:
                results['due_date'] = due_date
            elif len(dates) > 1 and results['invoice_date'] == dates[0]:
                # If no specific due date found but we have a second date, use it
                results['due_date'] = dates[1]
            
        # Extract names with better contextual understanding
        self.extract_names_with_context(text, results)
        
        # Extract monetary amounts
        amounts = self.extract_amounts(text)
        
        # Assign monetary values
        total = self.extract_field(text, 'total_amount')
        if total:
            results['total_amount'] = total
        elif amounts:
            # If no clear total found, use the largest amount
            results['total_amount'] = amounts[0]
            
        # Try to extract subtotal and tax
        results['subtotal'] = self.extract_field(text, 'subtotal')
        results['tax'] = self.extract_field(text, 'tax')
        
        # If we have a total but no subtotal, check if there's a secondary amount
        if results['total_amount'] and not results['subtotal'] and len(amounts) > 1:
            # Use second largest amount as subtotal
            results['subtotal'] = amounts[1]
            
        # Extract remaining fields
        results['email'] = self.extract_field(text, 'email')
        results['phone_number'] = self.extract_field(text, 'phone_number')
        results['payment_terms'] = self.extract_field(text, 'payment_terms')
        
        # Check for phone pattern directly if not found
        if not results['phone_number']:
            phone_match = re.search(r'(\d{3}[-\.\s]\d{3}[-\.\s]\d{4})', text)
            if phone_match:
                results['phone_number'] = phone_match.group(1)
        
        # Clean up any remaining None values for JSON serialization
        for key, value in results.items():
            if value is None:
                results[key] = ""
        
        return results
    
    def extract_names_with_context(self, text, results):
        """Extract issuer and recipient names with better context awareness"""
        # Split text into lines for better analysis
        lines = text.split('\n')
        
        # Clean lines and remove empty ones
        lines = [line.strip() for line in lines if line.strip()]
        
        # Find potential names and their contexts
        billed_to_idx = -1
        payment_info_idx = -1
        account_name_idx = -1
        
        for i, line in enumerate(lines):
            lower_line = line.lower()
            if 'billed to' in lower_line:
                billed_to_idx = i
            elif 'payment information' in lower_line:
                payment_info_idx = i
            elif 'account name' in lower_line:
                account_name_idx = i
        
        # Extract recipient name - usually appears after "BILLED TO"
        if billed_to_idx >= 0 and billed_to_idx + 1 < len(lines):
            # Extract the next line after "BILLED TO"
            potential_recipient = lines[billed_to_idx + 1]
            
            # Clean it from any invoice numbers or dates
            potential_recipient = re.sub(r'invoice\s+no\.?\s*\d+', '', potential_recipient, flags=re.IGNORECASE)
            potential_recipient = re.sub(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', '', potential_recipient, flags=re.IGNORECASE)
            potential_recipient = re.sub(r'\d{1,2}[-\.\/]\d{1,2}[-\.\/]\d{2,4}', '', potential_recipient)
            
            # Further clean and validate
            potential_recipient = potential_recipient.strip()
            if len(potential_recipient) > 2 and not self.is_header_text(potential_recipient):
                results['recipient_name'] = potential_recipient
        
        # If we couldn't find a recipient name that way, try to find a person's name
        if not results['recipient_name']:
            # Look for names near "BILLED TO"
            billed_to_area = ""
            if billed_to_idx >= 0:
                # Check a few lines after "BILLED TO"
                for j in range(billed_to_idx, min(billed_to_idx + 3, len(lines))):
                    billed_to_area += lines[j] + " "
            
            # Extract person names in this area
            name_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', billed_to_area)
            if name_match:
                name = name_match.group(1)
                if not self.is_header_text(name):
                    results['recipient_name'] = name
        
        # Extract issuer name - usually appears in payment information, account name, or at the end
        if account_name_idx >= 0 and account_name_idx + 1 < len(lines):
            # Extract from account name line
            account_name_line = lines[account_name_idx]
            account_name_match = re.search(r'account\s+name\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})', account_name_line, re.IGNORECASE)
            
            if account_name_match:
                results['issuer_name'] = account_name_match.group(1).strip()
            else:
                # Try to find name after the colon
                parts = account_name_line.split(':')
                if len(parts) > 1:
                    name_part = parts[1].strip()
                    name_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})', name_part)
                    if name_match:
                        results['issuer_name'] = name_match.group(1).strip()
        
        # If still no issuer name, look for names in the payment section
        if not results['issuer_name'] and payment_info_idx >= 0:
            payment_area = ""
            # Check several lines after payment information
            for j in range(payment_info_idx, min(payment_info_idx + 6, len(lines))):
                payment_area += lines[j] + " "
            
            # Look for names in payment area
            name_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', payment_area)
            for name in name_matches:
                if name != results['recipient_name'] and not self.is_header_text(name):
                    results['issuer_name'] = name
                    break
        
        # If we still don't have an issuer, try to find names at the document end
        if not results['issuer_name']:
            # Look at last 5 lines
            last_section = " ".join(lines[-5:])
            name_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', last_section)
            for name in name_matches:
                if name != results['recipient_name'] and not self.is_header_text(name):
                    results['issuer_name'] = name
                    break
        
        # Final fallback - check if we have names in standard issuer/recipient patterns
        if not results['issuer_name']:
            results['issuer_name'] = self.extract_field(text, 'issuer_name')
            
        if not results['recipient_name']:
            results['recipient_name'] = self.extract_field(text, 'recipient_name')
    
    def save_results(self, extracted_info, output_path):
        """Save the extracted information to a JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(extracted_info, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving extraction results: {e}")
            return False