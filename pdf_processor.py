import os, re, json, fitz, pytesseract
from pathlib import Path
from pdf2image import convert_from_path
from email.header import decode_header
from PIL import Image, ImageOps

# Configuration - Update these paths
SAVED_PATHS_FILE = Path("e:\\ProjektyGITHUB\\Grupa\\Projekt-Firma\\Data\\SavedPaths.json")

def load_saved_paths():
    """Load saved paths from the JSON file."""
    if SAVED_PATHS_FILE.exists():
        with open(SAVED_PATHS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_paths(paths):
    """Save paths to the JSON file."""
    SAVED_PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVED_PATHS_FILE, 'w') as f:
        json.dump(paths, f, indent=4)

def get_path_from_user(prompt):
    """Prompt the user to input a path."""
    while True:
        path = input(prompt).strip()
        if os.path.exists(path):
            return path
        print("Invalid path. Please try again.")

def setup_paths():
    """Ensure Tesseract and Poppler paths are set up."""
    paths = load_saved_paths()

    if "PYTESSERACT_PATH" not in paths or not os.path.exists(paths["PYTESSERACT_PATH"]):
        paths["PYTESSERACT_PATH"] = get_path_from_user("Enter the full path to Tesseract executable: ")

    if "POPPLER_PATH" not in paths or not os.path.exists(paths["POPPLER_PATH"]):
        paths["POPPLER_PATH"] = get_path_from_user("Enter the full path to Poppler bin directory: ")

    save_paths(paths)
    return paths

# Load or prompt for paths
paths = setup_paths()
PYTESSERACT_PATH = paths["PYTESSERACT_PATH"]
POPPLER_PATH = paths["POPPLER_PATH"]

# Configure environment
def setup_environment():
    os.environ['PATH'] += os.pathsep + POPPLER_PATH
    pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_PATH

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", str(filename))

def preprocess_image(img):
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    new_width = int(img.width * 2)
    new_height = int(img.height * 2)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img = img.point(lambda x: 0 if x < 180 else 255, '1')
    return img

def pdf_to_text(pdf_path):
    try:
        # Try direct text extraction first
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
            if len(text.strip()) > 50:  # If we got reasonable text
                return text

        # Fallback to OCR for scanned PDFs
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=500)
        processed_text = []

        for img in images:
            # Preprocess the image
            img = preprocess_image(img)

            # Perform OCR with Dutch language model
            custom_config = r'--oem 3 --psm 6 -l nld --dpi 500 -c preserve_interword_spaces=1 -c textord_debug_tabfind=0 ' # Use Dutch language model
            text = pytesseract.image_to_string(img, config=custom_config)
            processed_text.append(text)

        return "\n".join(processed_text)

    except Exception as e:
        print(f"PDF processing error for {pdf_path.name}: {str(e)}")
        return ""

def process_pdf_attachment(part, email_id, attachments_dir):
    try:
        filename = part.get_filename()
        if not filename or not filename.lower().endswith('.pdf'):
            return None  # Skip non-PDF files

        # Decode filename if needed
        filename_header = decode_header(filename)[0]
        if isinstance(filename_header[0], bytes):
            filename = filename_header[0].decode(filename_header[1] or 'utf-8')

        # Sanitize the filename and email_id to remove invalid characters
        sanitized_name = sanitize_filename(filename)
        sanitized_email_id = sanitize_filename(email_id)
        save_path = Path(attachments_dir) / f"{sanitized_email_id}_{sanitized_name}"

        # Save the PDF file
        with open(save_path, 'wb') as f:
            f.write(part.get_payload(decode=True))
        print(f"Saved PDF attachment: {save_path.name}")

        # Extract text from the PDF
        extracted_text = pdf_to_text(save_path)
        if extracted_text:
            print(f"Extracted text from {save_path.name}")
        try:
            os.remove(save_path) 
            print(f"Deleted temporary file: {save_path.name}") 
        except Exception as e:
            print(f"Error deleting temporary file {save_path.name}: {str(e)}")  
            
        return extracted_text

    except Exception as e:
        print(f"Error processing PDF attachment {filename}: {str(e)}")
        return None