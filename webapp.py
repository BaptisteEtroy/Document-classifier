from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import json
import io
import logging
import glob
import datetime
from werkzeug.utils import secure_filename
from classification.document_classifier import DocumentClassifier
from extraction.invoice_extractor import InvoiceExtractor

# Set up logging
log_stream = io.StringIO()
log_handler = logging.StreamHandler(log_stream)
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)

# Add handler to root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(log_handler)

# Also add console handler for terminal output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'txt', 'html', 'md'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize classifier and extractor
classifier = DocumentClassifier(models_dir='models')
extractor = InvoiceExtractor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Technical dashboard with system information and logs"""
    # Collect system information
    system_info = get_system_info()
    return render_template('dashboard.html', system_info=system_info)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/classify', methods=['POST'])
def classify_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Clear previous logs
        log_stream.truncate(0)
        log_stream.seek(0)
        
        logging.info(f"Processing file: {file.filename}")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logging.info(f"File saved to: {filepath}")
        
        # Classify document
        try:
            logging.info("Starting document classification...")
            category = classifier.classify(filepath)
            logging.info(f"Document classified as: {category}")
            
            result = {
                'filename': filename, 
                'category': category,
                'file_url': f'/uploads/{filename}'
            }
            
            # If document is an invoice, offer extraction
            if category == 'invoices':
                result['can_extract'] = True
                logging.info("Document is an invoice. Extraction available.")
            else:
                result['can_extract'] = False
                logging.info("Document is not an invoice. Extraction unavailable.")
            
            return jsonify(result)
        except Exception as e:
            logging.error(f"Error classifying document: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    logging.warning(f"File type not allowed: {file.filename}")
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/extract', methods=['POST'])
def extract_info():
    # Get the filename from the request
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    
    logging.info(f"Starting extraction for: {filename}")
    
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return jsonify({'error': 'File not found'}), 404
    
    # Extract information
    try:
        logging.info("Extracting information from invoice...")
        extracted_info = extractor.extract(filepath)
        logging.info("Extraction complete.")
        return jsonify(extracted_info)
    except Exception as e:
        logging.error(f"Error extracting information: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    logs = log_stream.getvalue()
    return jsonify({'logs': logs})

@app.route('/api/system-info', methods=['GET'])
def api_system_info():
    """API endpoint to get system information"""
    return jsonify(get_system_info())

def get_system_info():
    """Collect system information for the dashboard"""
    info = {
        'models': get_models_info(),
        'training_data': get_training_data_info(),
        'recent_uploads': get_recent_uploads(),
        'system_status': {
            'app_start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': os.popen('python --version').read().strip(),
            'memory_usage': os.popen('ps -o rss= -p ' + str(os.getpid())).read().strip() + ' KB' if os.name != 'nt' else 'N/A',
            'cpu_usage': os.popen('ps -o %cpu= -p ' + str(os.getpid())).read().strip() + '%' if os.name != 'nt' else 'N/A'
        }
    }
    return info

def get_models_info():
    """Get information about the trained models"""
    models_dir = 'models'
    models = []
    
    # Check for model files
    model_files = glob.glob(os.path.join(models_dir, '*.pkl'))
    
    for model_file in model_files:
        model_name = os.path.basename(model_file)
        model_size = os.path.getsize(model_file)
        model_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_file))
        
        models.append({
            'name': model_name,
            'size': format_file_size(model_size),
            'last_modified': model_time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return models

def get_training_data_info():
    """Get information about the training data"""
    data_dir = 'data'
    data_info = {}
    
    if not os.path.exists(data_dir):
        return {'error': 'Data directory not found'}
    
    categories = ['invoices', 'Shipping orders', 'resumes', 'others']
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        if os.path.exists(category_path):
            files = glob.glob(os.path.join(category_path, '*.*'))
            
            # Group files by extension
            extensions = {}
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    extensions[ext] += 1
                else:
                    extensions[ext] = 1
            
            data_info[category] = {
                'total_files': len(files),
                'extensions': extensions
            }
    
    return data_info

def get_recent_uploads():
    """Get information about recent uploads"""
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        return []
    
    files = glob.glob(os.path.join(uploads_dir, '*.*'))
    
    # Sort files by modification time, newest first
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    recent_files = []
    for file in files[:10]:  # Limit to 10 most recent files
        filename = os.path.basename(file)
        file_size = os.path.getsize(file)
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file))
        
        recent_files.append({
            'name': filename,
            'size': format_file_size(file_size),
            'uploaded': file_time.strftime('%Y-%m-%d %H:%M:%S'),
        })
    
    return recent_files

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

if __name__ == '__main__':
    app.run(debug=True) 