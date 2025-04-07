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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('app.log')  # Log to file
    ]
)

# Get the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'txt', 'html', 'md'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

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
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"Processing file: {filename}")
        logger.info(f"File saved to: {file_path}")
        
        # Classify the document
        logger.info("Starting document classification...")
        category = classifier.classify(file_path)
        logger.info(f"Document classified as: {category}")
        
        # Determine if extraction is available
        extraction_available = (category == 'invoices')
        if extraction_available:
            logger.info("Document is an invoice. Extraction available.")
        
        # Return the result
        return jsonify({
            'success': True,
            'file': filename,
            'category': category,
            'extraction_available': extraction_available
        })
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/extract', methods=['POST'])
def extract_info():
    # Get filename from request
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'})
    
    filename = data['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'})
    
    # Extract information
    logger.info(f"Starting extraction for: {filename}")
    logger.info("Extracting information from invoice...")
    extracted_info = extractor.extract(file_path)
    
    # Save the extraction result
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}_extracted.json")
    extractor.save_results(extracted_info, output_path)
    
    logger.info("Extraction complete.")
    return jsonify(extracted_info)

@app.route('/logs', methods=['GET'])
def get_logs():
    """Endpoint to retrieve recent logs for the dashboard"""
    try:
        with open('app.log', 'r') as log_file:
            logs = log_file.readlines()
            # Clean up the logs (remove newlines, empty lines)
            logs = [log.strip() for log in logs if log.strip()]
            # Return the last 50 log entries or fewer if there aren't that many
            return jsonify({'logs': logs[-50:] if len(logs) > 50 else logs})
    except FileNotFoundError:
        return jsonify({'logs': ['No logs available yet.']})

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