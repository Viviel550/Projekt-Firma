#!/usr/bin/env python3
# main.py - Główny skrypt uruchamiający PDF Email Extractor

import os
import sys
import argparse
import json
import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict, Any

# Dodajemy bieżący katalog do ścieżki Pythona
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Importuj moduły z nowej struktury
try:
    # Import from modular structure
    from extractor import EnhancedDataExtractor
    from data_validator import DataValidator
    
    # Try to import config, create default if missing
    try:
        import config
        from config import *
    except ImportError:
        print("Warning: config.py not found, using default settings...")
        # Default settings
        INPUT_PDF_DIR = "pdfs"
        OUTPUT_DIR = "output"
        OUTPUT_EXCEL_FILE = "extracted_data.xlsx"
        LOGGING_CONFIG = {
            'level': 'INFO',
            'log_file': 'logs/extractor.log',
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
        PERFORMANCE_SETTINGS = {
            'max_pdf_size_mb': 50
        }
        # Create directories
        for directory in [INPUT_PDF_DIR, OUTPUT_DIR, "logs"]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
except ImportError as e:
    print(f"Błąd importu: {e}")
    print("Upewnij się, że wszystkie wymagane pliki znajdują się w katalogu projektu.")
    print("Uruchom 'pip install -r requirements.txt' aby zainstalować wymagane biblioteki.")
    sys.exit(1)

# Konfiguracja logowania
def setup_logging():
    """Konfiguruje system logowania"""
    log_dir = os.path.dirname(LOGGING_CONFIG['log_file'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['log_format'],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG['log_file']),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class ExtractorManager:
    """Główna klasa zarządzająca ekstraktorem"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.extractor = EnhancedDataExtractor()
        self.validator = DataValidator()
        self.stats = {
            'processed_files': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'start_time': datetime.now()
        }
    
    def process_single_pdf(self, file_path: str) -> Dict[str, Any]:
        """Przetwarza pojedynczy plik PDF"""
        self.logger.info(f"Przetwarzanie pliku: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Plik nie istnieje: {file_path}")
            
            # Sprawdź rozmiar pliku
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > PERFORMANCE_SETTINGS['max_pdf_size_mb']:
                raise ValueError(f"Plik zbyt duży: {file_size_mb:.1f}MB")
            
            extracted_data = self.extractor.extract_from_pdf(file_path)
            extracted_data['source_file'] = os.path.basename(file_path)
            extracted_data['file_size_mb'] = file_size_mb
            
            # Walidacja danych za pomocą DataValidator
            validation_result = self.validator.validate_extracted_data(extracted_data)
            extracted_data['validation'] = validation_result
            
            self.stats['successful_extractions'] += 1
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Błąd przetwarzania {file_path}: {str(e)}")
            self.stats['failed_extractions'] += 1
            return {
                'source_file': os.path.basename(file_path),
                'error': str(e),
                'validation': {'is_valid': False, 'errors': [str(e)]}
            }
        finally:
            self.stats['processed_files'] += 1
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Przetwarza wszystkie pliki PDF w katalogu"""
        self.logger.info(f"Przetwarzanie katalogu: {directory_path}")
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            self.logger.warning(f"Utworzono katalog: {directory_path}")
            return []
        
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            self.logger.warning("Nie znaleziono plików PDF w katalogu")
            return []
        
        self.logger.info(f"Znaleziono {len(pdf_files)} plików PDF")
        
        all_data = []
        for filename in pdf_files:
            file_path = os.path.join(directory_path, filename)
            extracted_data = self.process_single_pdf(file_path)
            all_data.append(extracted_data)
        
        return all_data
    
    def process_emails(self, email_config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Przetwarza tylko nieprzeczytane emaile z serwera IMAP i oznacza je jako przeczytane"""
        self.logger.info("Rozpoczynam przetwarzanie nieprzeczytanych emaili")
        
        try:
            # Update the call to specify unread_only=True
            emails_data = self.extractor.read_emails_from_imap(
                server=email_config['server'],
                username=email_config['username'],
                password=email_config['password'],
                folder=email_config.get('folder', 'INBOX'),
                unread_only=True,  # Only process unread emails
                mark_as_read=True  # Mark emails as read after processing
            )
            
            # Walidacja danych z emaili
            for email_data in emails_data:
                validation_result = self.validator.validate_extracted_data(email_data)
                email_data['validation'] = validation_result
            
            self.logger.info(f"Przetworzono {len(emails_data)} nieprzeczytanych emaili")
            return emails_data
            
        except Exception as e:
            self.logger.error(f"Błąd przetwarzania emaili: {str(e)}")
            return []
    
    def export_to_excel(self, data: List[Dict[str, Any]], output_file: str) -> bool:
        """Eksportuje dane do Excel z zaawansowanym formatowaniem"""
        try:
            # Upewnij się, że katalog wyjściowy istnieje
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Zapisz dane używając metody z ExtractorManager
            success = self.extractor.save_to_excel(data, output_file)
            
            if success:
                # Dodaj arkusz statystyk
                self.add_statistics_sheet(output_file, data)
                self.logger.info(f"Dane wyeksportowane do: {output_file}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Błąd eksportu do Excel: {str(e)}")
            return False
    
    def add_statistics_sheet(self, excel_file: str, data: List[Dict[str, Any]]):
        """Dodaje arkusz ze statystykami"""
        try:
            # Przygotuj statystyki
            total_records = len(data)
            valid_records = sum(1 for d in data if d.get('validation', {}).get('is_valid', True))
            invalid_records = total_records - valid_records
            
            processing_time = datetime.now() - self.stats['start_time']
            
            stats_data = {
                'Statystyka': [
                    'Łączna liczba rekordów',
                    'Poprawne rekordy',
                    'Niepoprawne rekordy',
                    'Procent sukcesu',
                    'Czas przetwarzania',
                    'Data generowania raportu'
                ],
                'Wartość': [
                    total_records,
                    valid_records,
                    invalid_records,
                    f"{(valid_records/total_records*100):.1f}%" if total_records > 0 else "0%",
                    str(processing_time).split('.')[0],
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            
            # Dodaj arkusz statystyk
            try:
                with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
                    pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statystyki', index=False)
            except Exception as e:
                self.logger.warning(f"Nie udało się dodać arkusza statystyk: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Błąd dodawania arkusza statystyk: {str(e)}")
    
    def export_to_json(self, data: List[Dict[str, Any]], output_file: str) -> bool:
        """Eksportuje dane do JSON"""
        return self.extractor.export_to_json(data, output_file)
    
    def generate_report(self, data: List[Dict[str, Any]]) -> str:
        """Generuje raport z przetwarzania"""
        total_records = len(data)
        valid_records = sum(1 for d in data if d.get('validation', {}).get('is_valid', True))
        
        report = f"""
=== RAPORT PRZETWARZANIA ===
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Łączna liczba rekordów: {total_records}
Poprawne rekordy: {valid_records}
Niepoprawne rekordy: {total_records - valid_records}
Procent sukcesu: {(valid_records/total_records*100):.1f}%

=== STATYSTYKI PLIKÓW ===
Przetworzone pliki: {self.stats['processed_files']}
Udane ekstrakcje: {self.stats['successful_extractions']}
Nieudane ekstrakcje: {self.stats['failed_extractions']}

=== BRAKUJĄCE DANE ===
"""
        
        # Analiza brakujących pól
        missing_fields = {}
        for record in data:
            if record.get('validation'):
                for field in record['validation'].get('missing_required_fields', []):
                    missing_fields[field] = missing_fields.get(field, 0) + 1
        
        if missing_fields:
            for field, count in sorted(missing_fields.items(), key=lambda x: x[1], reverse=True):
                report += f"{field}: {count} razy\n"
        else:
            report += "Brak brakujących wymaganych pól.\n"
        
        return report

def parse_arguments():
    """Parsuje argumenty wiersza poleceń"""
    parser = argparse.ArgumentParser(description='PDF Email Data Extractor')
    parser.add_argument('--mode', choices=['pdf', 'email', 'single', 'interactive'], 
                       default='interactive', help='Tryb przetwarzania')
    parser.add_argument('--input', help='Ścieżka do pliku lub katalogu wejściowego')
    parser.add_argument('--output', help='Ścieżka do pliku wyjściowego')
    parser.add_argument('--format', choices=['excel', 'json', 'both'], 
                       default='excel', help='Format wyjściowy')
    parser.add_argument('--config', help='Ścieżka do pliku konfiguracyjnego')
    parser.add_argument('--verbose', '-v', action='store_true', help='Szczegółowe logowanie')
    
    return parser.parse_args()

def load_email_credentials():
    """Ładuje dane dostępowe do emaila z pliku .env lub z inputu"""
    email_config = {}
    
    # Spróbuj załadować z pliku .env
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key.startswith('EMAIL_'):
                        email_config[key.lower().replace('email_', '')] = value
    
    # Jeśli brak danych, poproś użytkownika
    if not email_config.get('server'):
        print("\n=== Konfiguracja emaila ===")
        email_config['server'] = input("Serwer IMAP (np. imap.gmail.com): ")
        email_config['username'] = input("Email: ")
        email_config['password'] = input("Hasło: ")
        email_config['folder'] = input("Folder (domyślnie INBOX): ") or 'INBOX'
    
    return email_config

def interactive_mode():
    """Tryb interaktywny"""
    manager = ExtractorManager()
    
    while True:
        print("\n=== PDF Email Data Extractor ===")
        print("1. Przetwarzaj pliki PDF z katalogu")
        print("2. Przetwarzaj pojedynczy plik PDF")
        print("3. Przetwarzaj emaile")
        print("4. Przetwarzaj pojedynczy email (tekst)")
        print("5. Trenuj model ML")
        print("6. Eksportuj dane do JSON")
        print("7. Ewaluacja jakości ekstrakcji")
        print("8. Pokaż statystyki")
        print("9. Wyjście")
        
        choice = input("\nWybierz opcję (1-9): ").strip()
        
        if choice == '1':
            input_dir = input(f"Katalog wejściowy (domyślnie {INPUT_PDF_DIR}): ") or INPUT_PDF_DIR
            data = manager.process_directory(input_dir)
            
            if data:
                output_file = input(f"Plik wyjściowy (domyślnie {OUTPUT_EXCEL_FILE}): ") or OUTPUT_EXCEL_FILE
                output_path = os.path.join(OUTPUT_DIR, output_file)
                
                if manager.export_to_excel(data, output_path):
                    print(f"\n✓ Przetwarzanie zakończone. Wyniki w: {output_path}")
                    print(manager.generate_report(data))
        
        elif choice == '2':
            file_path = input("Ścieżka do pliku PDF: ").strip()
            if file_path:
                data = [manager.process_single_pdf(file_path)]
                output_path = os.path.join(OUTPUT_DIR, f"single_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                
                if manager.export_to_excel(data, output_path):
                    print(f"\n✓ Plik przetworzony. Wyniki w: {output_path}")
        
        elif choice == '3':
            email_config = load_email_credentials()
            data = manager.process_emails(email_config)
            
            if data:
                output_path = os.path.join(OUTPUT_DIR, f"emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                if manager.export_to_excel(data, output_path):
                    print(f"\n✓ Emaile przetworzone. Wyniki w: {output_path}")
                    
                    # Pokaż dodatkowe statystyki dla emaili
                    total_items = 0
                    for email_data in data:
                        if email_data.get('material_descriptions_list'):
                            total_items += len(email_data['material_descriptions_list'])
                        else:
                            total_items += 1
                    print(f"Łączna liczba pozycji w Excel: {total_items}")
        
        elif choice == '4':
            # Przetwarzanie pojedynczego emaila z pliku tekstowego
            txt_path = input("Podaj ścieżkę do pliku tekstowego z emailem: ").strip()
            
            if not txt_path:
                print("Nie podano ścieżki do pliku")
                continue
            
            if not os.path.exists(txt_path):
                print(f"Plik nie istnieje: {txt_path}")
                continue
            
            try:
                # Wczytaj treść z pliku tekstowego
                with open(txt_path, 'r', encoding='utf-8') as f:
                    email_content = f.read()
                
                if not email_content.strip():
                    print("Plik jest pusty")
                    continue
                
                print(f"Wczytano treść emaila z pliku: {txt_path}")
                print(f"Długość tekstu: {len(email_content)} znaków")
                
                # Użyj nowego modułu extractor
                extracted_data = manager.extractor.extract_from_email(email_content)
                extracted_data['source_file'] = os.path.basename(txt_path)
                
                # Walidacja danych z użyciem DataValidator
                validation_result = manager.validator.validate_extracted_data(extracted_data)
                extracted_data['validation'] = validation_result
                
                # Pokaż statystyki jakości
                quality = manager.validator.evaluate_quality(extracted_data)
                print(f"\nJakość ekstrakcji:")
                print(f"- Kompletność: {quality['completeness']*100:.1f}%")
                print(f"- Średnia pewność: {quality['confidence']*100:.1f}%")
                print(f"- Wszystkie wymagane pola: {'Tak' if quality['required_fields_present'] else 'Nie'}")
                
                if quality['warnings']:
                    print("Ostrzeżenia:")
                    for warning in quality['warnings']:
                        print(f"  - {warning}")
                
                if validation_result['errors']:
                    print("Błędy walidacji:")
                    for error in validation_result['errors']:
                        print(f"  - {error}")
                
                # Zapisz do Excel
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(OUTPUT_DIR, f"email_from_file_{timestamp}.xlsx")
                
                if manager.export_to_excel([extracted_data], output_path):
                    print(f"\n✓ Przetwarzanie zakończone. Wyniki zapisane w: {output_path}")
                    
                    # Pokaż ile wierszy zostanie utworzonych
                    if extracted_data.get('material_descriptions_list'):
                        print(f"Utworzono {len(extracted_data['material_descriptions_list'])} wierszy dla pozycji materiałów")
                    else:
                        print("Utworzono 1 wiersz")
                else:
                    print("Błąd podczas przetwarzania")
                    
                # Pokaż wyekstrahowane dane
                print("\n=== WYEKSTRAHOWANE DANE ===")
                for key, value in extracted_data.items():
                    if key not in ['order_items', 'material_descriptions_list', 'validation']:
                        print(f"{key}: {value}")
                
                if extracted_data.get('material_descriptions_list'):
                    print(f"\nZnalezione materiały ({len(extracted_data['material_descriptions_list'])}):")
                    for i, desc in enumerate(extracted_data['material_descriptions_list'], 1):
                        print(f"  {i}. {desc}")
                        
                # Opcjonalnie pokaż fragment oryginalnego tekstu
                preview_choice = input("\nCzy chcesz zobaczyć fragment oryginalnego tekstu? (t/n): ").strip().lower()
                if preview_choice == 't':
                    preview_length = min(500, len(email_content))
                    print(f"\nFragment tekstu (pierwsze {preview_length} znaków):")
                    print("-" * 50)
                    print(email_content[:preview_length])
                    if len(email_content) > preview_length:
                        print("...")
                    print("-" * 50)
            
            except UnicodeDecodeError:
                print("Błąd dekodowania pliku. Spróbuj z innym kodowaniem.")
                try:
                    with open(txt_path, 'r', encoding='latin-1') as f:
                        email_content = f.read()
                    print("Pomyślnie wczytano plik z kodowaniem latin-1")
                    # Continue with processing...
                except Exception as e:
                    print(f"Nie udało się wczytać pliku: {str(e)}")
            
            except Exception as e:
                print(f"Błąd podczas wczytywania pliku: {str(e)}")
        
        elif choice == '5':
            print("Trenowanie modelu ML...")
            try:
                # Access the traditional extractor for training
                if hasattr(manager.extractor, 'traditional_extractor'):
                    manager.extractor.traditional_extractor.train_model()
                    print("\n✓ Model ML został wytrenowany i zapisany")
                else:
                    print("Model ML nie jest dostępny w obecnej konfiguracji")
            except Exception as e:
                print(f"Błąd podczas trenowania modelu: {str(e)}")
        
        elif choice == '6':
            # Eksport do JSON
            print("Dostępne pliki Excel:")
            excel_files = []
            if os.path.exists(OUTPUT_DIR):
                for file in os.listdir(OUTPUT_DIR):
                    if file.endswith('.xlsx'):
                        excel_files.append(file)
                        print(f"{len(excel_files)}. {file}")
            
            if excel_files:
                try:
                    file_choice = input("Wybierz numer pliku lub podaj pełną ścieżkę: ").strip()
                    
                    if file_choice.isdigit() and 1 <= int(file_choice) <= len(excel_files):
                        excel_file = os.path.join(OUTPUT_DIR, excel_files[int(file_choice) - 1])
                    else:
                        excel_file = file_choice
                    
                    if os.path.exists(excel_file):
                        json_file = excel_file.replace('.xlsx', '.json')
                        
                        # Wczytaj dane z Excel i zapisz jako JSON
                        df = pd.read_excel(excel_file)
                        data_list = df.to_dict('records')
                        
                        if manager.export_to_json(data_list, json_file):
                            print(f"\n✓ Dane zostały wyeksportowane do: {json_file}")
                        else:
                            print("Błąd podczas eksportu do JSON")
                    else:
                        print("Plik nie istnieje")
                except Exception as e:
                    print(f"Błąd podczas eksportu: {str(e)}")
            else:
                print("Brak plików Excel do eksportu")
        
        elif choice == '7':
            # Ewaluacja jakości ekstrakcji
            print("Wybierz opcję ewaluacji:")
            print("1. Ewaluacja pliku PDF")
            print("2. Ewaluacja tekstu emaila")
            
            eval_choice = input("Wybierz opcję (1-2): ").strip()
            
            if eval_choice == '1':
                pdf_path = input("Podaj ścieżkę do pliku PDF: ").strip()
                if os.path.exists(pdf_path):
                    print(f"Ewaluacja jakości ekstrakcji dla: {pdf_path}")
                    extracted_data = manager.extractor.extract_from_pdf(pdf_path)
                    
                    # Walidacja z DataValidator
                    validation_result = manager.validator.validate_extracted_data(extracted_data)
                    extracted_data['validation'] = validation_result
                    
                    quality = manager.validator.evaluate_quality(extracted_data)
                    
                    print("\n=== STATYSTYKI JAKOŚCI EKSTRAKCJI ===")
                    print(f"Kompletność: {quality['completeness']*100:.1f}%")
                    print(f"Średnia pewność: {quality['confidence']*100:.1f}%")
                    print(f"Wszystkie wymagane pola: {'Tak' if quality['required_fields_present'] else 'Nie'}")
                    
                    if quality['warnings']:
                        print("\nOstrzeżenia:")
                        for warning in quality['warnings']:
                            print(f"  - {warning}")
                    
                    if validation_result['errors']:
                        print("\nBłędy walidacji:")
                        for error in validation_result['errors']:
                            print(f"  - {error}")
                    
                    # Pokaż wyekstrahowane dane
                    print("\n=== WYEKSTRAHOWANE DANE ===")
                    for key, value in extracted_data.items():
                        if key not in ['order_items', 'validation'] and isinstance(value, (str, int, float, bool)):
                            print(f"{key}: {value}")
                else:
                    print("Plik nie istnieje")
            
            elif eval_choice == '2':
                print("Wklej treść emaila (zakończ wpisując 'END' w nowej linii):")
                email_lines = []
                while True:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    email_lines.append(line)
                
                email_content = "\n".join(email_lines)
                
                if email_content.strip():
                    extracted_data = manager.extractor.extract_from_email(email_content)
                    validation_result = manager.validator.validate_extracted_data(extracted_data)
                    
                    quality = manager.validator.evaluate_quality(extracted_data)
                    
                    print("\n=== STATYSTYKI JAKOŚCI EKSTRAKCJI EMAILA ===")
                    print(f"Typ dokumentu: {extracted_data.get('email_type', 'unknown')}")
                    print(f"Kompletność: {quality['completeness']*100:.1f}%")
                    print(f"Średnia pewność: {quality['confidence']*100:.1f}%")
                    print(f"Wszystkie wymagane pola: {'Tak' if quality['required_fields_present'] else 'Nie'}")
                    
                    if quality['warnings']:
                        print("\nOstrzeżenia:")
                        for warning in quality['warnings']:
                            print(f"  - {warning}")
                    
                    print("\n=== WYEKSTRAHOWANE DANE ===")
                    for key, value in extracted_data.items():
                        if key not in ['order_items', 'material_descriptions_list']:
                            print(f"{key}: {value}")
                    
                    if extracted_data.get('material_descriptions_list'):
                        print(f"\nZnalezione materiały ({len(extracted_data['material_descriptions_list'])}):")
                        for i, desc in enumerate(extracted_data['material_descriptions_list'], 1):
                            print(f"  {i}. {desc}")
                else:
                    print("Brak treści emaila")
        
        elif choice == '8':
            print(f"\n=== STATYSTYKI SESJI ===")
            print(f"Przetworzone pliki: {manager.stats['processed_files']}")
            print(f"Udane ekstrakcje: {manager.stats['successful_extractions']}")
            print(f"Nieudane ekstrakcje: {manager.stats['failed_extractions']}")
            print(f"Czas działania: {datetime.now() - manager.stats['start_time']}")
            
            # Sprawdź dostępne pliki wynikowe
            if os.path.exists(OUTPUT_DIR):
                output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.xlsx', '.json'))]
                if output_files:
                    print(f"\nDostępne pliki wynikowe ({len(output_files)}):")
                    for file in sorted(output_files):
                        file_path = os.path.join(OUTPUT_DIR, file)
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        print(f"  - {file} ({file_size:.1f} KB, {mod_time.strftime('%Y-%m-%d %H:%M')})")
        
        elif choice == '9':
            print("Do widzenia!")
            break
        
        else:
            print("Nieprawidłowy wybór! Wybierz opcję od 1 do 9.")

def main():
    """Główna funkcja programu"""
    args = parse_arguments()
    
    # Ustawienia logowania
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.mode == 'interactive':
        interactive_mode()
    else:
        manager = ExtractorManager()
        data = []
        
        if args.mode == 'pdf':
            if args.input:
                if os.path.isdir(args.input):
                    data = manager.process_directory(args.input)
                else:
                    data = [manager.process_single_pdf(args.input)]
            else:
                data = manager.process_directory(INPUT_PDF_DIR)
        
        elif args.mode == 'email':
            email_config = load_email_credentials()
            data = manager.process_emails(email_config)
        
        elif args.mode == 'single':
            if not args.input:
                print("Błąd: Wymagana ścieżka do pliku (--input)")
                sys.exit(1)
            data = [manager.process_single_pdf(args.input)]
        
        # Eksport wyników
        if data:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if args.format in ['excel', 'both']:
                excel_file = args.output or os.path.join(OUTPUT_DIR, f"results_{timestamp}.xlsx")
                manager.export_to_excel(data, excel_file)
            
            if args.format in ['json', 'both']:
                json_file = args.output or os.path.join(OUTPUT_DIR, f"results_{timestamp}.json")
                manager.export_to_json(data, json_file)
            
            # Wyświetl raport
            print(manager.generate_report(data))
        else:
            print("Brak danych do przetworzenia")

if __name__ == "__main__":
    main()