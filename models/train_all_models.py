import os
import sys

# Add the models directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from train_invoice_model import train_invoice_model
from train_shipping_orders_model import train_shipping_orders_model
from train_resume_model import train_resume_model

def train_all_models(data_dir='data'):
    """Train all document classification models"""
    print("=== Training Document Classification Models ===")
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Train the invoice model
    print("\n=== Training Invoice Model ===")
    train_invoice_model(data_dir=data_dir, output_model_path=os.path.join(models_dir, 'invoice_model.pkl'))
    
    # Train the shipping orders model
    print("\n=== Training Shipping Orders Model ===")
    train_shipping_orders_model(data_dir=data_dir, output_model_path=os.path.join(models_dir, 'shipping_orders_model.pkl'))
    
    # Train the resume model
    print("\n=== Training Resume Model ===")
    train_resume_model(data_dir=data_dir, output_model_path=os.path.join(models_dir, 'resume_model.pkl'))
    
    print("\n=== All models trained successfully ===")

if __name__ == "__main__":
    # Get data directory from command line argument if provided
    data_dir = 'data'
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    train_all_models(data_dir)