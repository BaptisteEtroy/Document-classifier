#!/usr/bin/env python3

import os
import sys
from train_invoice_model import train_invoice_model
from train_contract_model import train_contract_model
from train_medical_model import train_medical_model

def train_all_models(data_dir='app/data'):
    """Train all document classification models"""
    print("=== Training Document Classification Models ===")
    
    # Train the invoice model
    print("\n=== Training Invoice Model ===")
    train_invoice_model(data_dir=data_dir)
    
    # Train the contract model
    print("\n=== Training Contract Model ===")
    train_contract_model(data_dir=data_dir)
    
    # Train the medical model
    print("\n=== Training Medical Model ===")
    train_medical_model(data_dir=data_dir)
    
    print("\n=== All models trained successfully ===")

if __name__ == "__main__":
    # Get data directory from command line argument if provided
    data_dir = 'app/data'
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    train_all_models(data_dir)