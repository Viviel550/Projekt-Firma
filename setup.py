#!/usr/bin/env python3
# setup.py - instalacja wymaganych bibliotek i konfiguracja projektu

import os
import sys
import subprocess
import logging

def setup_logging():
    """Konfiguruje system logowania"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Sprawdza wersję Pythona"""
    logger.info(f"Używana wersja Pythona: {sys.version}")
    
    if sys.version_info < (3, 8):
        logger.error("Wymagana jest wersja Pythona >= 3.8")
        return False
    return True

def install_requirements():
    """Instaluje wymagane biblioteki z requirements.txt"""
    try:
        if not os.path.exists("requirements.txt"):
            logger.error("Brak pliku requirements.txt")
            return False
        
        logger.info("Instalacja wymaganych bibliotek...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Biblioteki zainstalowane pomyślnie")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Błąd podczas instalacji bibliotek: {str(e)}")
        return False

def create_directories():
    """Tworzy wymagane katalogi"""
    required_dirs = [
        "input_pdfs",
        "output",
        "logs",
        "models"
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Utworzono katalog: {directory}")
        else:
            logger.info(f"Katalog {directory} już istnieje")
    
    return True

def create_env_file():
    """Tworzy plik .env jeśli nie istnieje"""
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("""# Dane logowania do emaila - NIGDY NIE COMMITUJ TEGO PLIKU!
EMAIL_SERVER=imap.example.com
EMAIL_USERNAME=your_email@example.com
EMAIL_PASSWORD=your_password
EMAIL_FOLDER=INBOX
""")
        logger.info("Utworzono plik .env z przykładową konfiguracją")
    else:
        logger.info("Plik .env już istnieje")

def check_imports():
    """Sprawdza czy można zaimportować wymagane moduły"""
    try:
        import pandas
        import PyPDF2
        import pdfplumber
        import openpyxl
        import nltk
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Pobieranie niezbędnych zasobów NLTK
        nltk.download('punkt', quiet=True)
        
        logger.info("Wszystkie wymagane moduły zostały pomyślnie zaimportowane")
        return True
    except ImportError as e:
        logger.error(f"Błąd importu: {str(e)}")
        return False

def main():
    """Główna funkcja setup"""
    logger.info("=== Konfiguracja PDF Email Extractor ===")
    
    # Sprawdź wersję Pythona
    if not check_python_version():
        return False
    
    # Zainstaluj wymagane biblioteki
    if not install_requirements():
        return False
    
    # Utwórz katalogi
    create_directories()
    
    # Utwórz plik .env
    create_env_file()
    
    # Sprawdź czy importy działają
    if not check_imports():
        return False
    
    logger.info("=== Konfiguracja zakończona pomyślnie! ===")
    logger.info("Możesz teraz uruchomić program: python main.py")
    return True

if __name__ == "__main__":
    logger = setup_logging()
    success = main()
    sys.exit(0 if success else 1)