#!/usr/bin/env python3

import os
import argparse
import sys
from classification.document_classifier import DocumentClassifier
from extraction.invoice_extractor import InvoiceExtractor
from utils.file_utils import is_file_supported

def get_user_input():
    """Get input and output paths interactively from the user"""
    print("\n=== Document Classification and Information Extraction ===\n")
    
    # Get input path
    while True:
        input_path = input("Enter the path to the document or directory to analyze: ").strip()
        if not input_path:
            print("Path cannot be empty. Please try again.")
            continue
        
        if not os.path.exists(input_path):
            print(f"Path '{input_path}' does not exist. Please enter a valid path.")
            continue
        
        break
    
    # Get output directory
    output_dir = input("Enter the output directory (press Enter for default 'output'): ").strip()
    if not output_dir:
        output_dir = "output"
    
    return input_path, output_dir

def main():
    # Check if arguments were provided
    if len(sys.argv) > 1:
        # Use command-line argument parsing
        parser = argparse.ArgumentParser(description='Document Classification and Information Extraction')
        parser.add_argument('--input', '-i', type=str, help='Input file or directory')
        parser.add_argument('--output', '-o', type=str, default='output', help='Output directory')
        parser.add_argument('--models-dir', '-m', type=str, default='models', help='Directory containing trained models')
        args = parser.parse_args()
        
        input_path = args.input
        output_dir = args.output
        models_dir = args.models_dir
    else:
        # Interactive mode
        input_path, output_dir = get_user_input()
        models_dir = 'models'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize classifier
    print(f"Loading models from {models_dir}...")
    classifier = DocumentClassifier(models_dir=models_dir)
    
    # Initialize extractor
    extractor = InvoiceExtractor()
    
    # Process input
    if os.path.isfile(input_path):
        print(f"\nProcessing file: {input_path}")
        process_file(input_path, output_dir, classifier, extractor)
        print(f"\nResults saved to {output_dir}")
    elif os.path.isdir(input_path):
        files_processed = 0
        print(f"\nProcessing directory: {input_path}")
        for root, _, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                if is_file_supported(file_path):
                    process_file(file_path, output_dir, classifier, extractor)
                    files_processed += 1
        
        print(f"\nProcessed {files_processed} files. Results saved to {output_dir}")
    else:
        print(f"Error: {input_path} is not a valid file or directory")

def process_file(file_path, output_dir, classifier, extractor):
    """Process a single document: classify and extract information if it's an invoice"""
    print(f"Processing: {file_path}")
    
    try:
        # Classify document
        category = classifier.classify(file_path)
        print(f"Classified as: {category}")
        
        # Extract information if document is an invoice
        if category == "invoices":
            print("Extracting information from invoice...")
            extracted_info = extractor.extract(file_path)
            
            # Save extracted information
            base_filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"{base_filename}_extracted.json")
            extractor.save_results(extracted_info, output_path)
            print(f"Extracted information saved to: {output_path}")
        
        # Save classification result
        classification_path = classifier.save_result(file_path, category, output_dir)
        print(f"Classification result saved to: {classification_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main() 