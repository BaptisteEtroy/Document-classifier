import os
import shutil
import mimetypes

def get_file_type(file_path):
    """Determine file type based on extension or MIME type"""
    _, ext = os.path.splitext(file_path)
    
    # Common document types
    if ext.lower() in ['.pdf']:
        return 'pdf'
    elif ext.lower() in ['.doc', '.docx']:
        return 'word'
    elif ext.lower() in ['.xls', '.xlsx']:
        return 'excel'
    elif ext.lower() in ['.ppt', '.pptx']:
        return 'powerpoint'
    elif ext.lower() in ['.txt', '.md', '.text']:
        return 'text'
    elif ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
        return 'image'
    elif ext.lower() in ['.html', '.htm']:
        return 'html'
    
    # If extension doesn't provide enough information, use MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        main_type = mime_type.split('/')[0]
        if main_type == 'application':
            sub_type = mime_type.split('/')[1]
            if 'pdf' in sub_type:
                return 'pdf'
            elif 'word' in sub_type or 'document' in sub_type:
                return 'word'
            elif 'excel' in sub_type or 'sheet' in sub_type:
                return 'excel'
            elif 'powerpoint' in sub_type or 'presentation' in sub_type:
                return 'powerpoint'
        elif main_type == 'text':
            return 'text'
        elif main_type == 'image':
            return 'image'
    
    return 'unknown'

def create_output_directory(base_dir, category):
    """Create an output directory for a specific category"""
    output_dir = os.path.join(base_dir, category)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def copy_file_to_category(file_path, category, output_base):
    """Copy a file to the appropriate category directory"""
    category_dir = create_output_directory(output_base, category)
    dest_path = os.path.join(category_dir, os.path.basename(file_path))
    shutil.copy2(file_path, dest_path)
    return dest_path

def list_files_in_directory(directory, recursive=False):
    """List all files in a directory"""
    file_list = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                file_list.append(os.path.join(root, file))
    else:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                file_list.append(item_path)
    
    return file_list

def get_supported_extensions():
    """Get a list of supported file extensions"""
    return [
        '.pdf', '.doc', '.docx', '.txt', '.md', '.text',
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif',
        '.html', '.htm'
    ]

def is_file_supported(file_path):
    """Check if a file is supported by the system"""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in get_supported_extensions() 