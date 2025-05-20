# Automatyzowanie przyjmowania zamówień ML

## Opis projektu

Aplikacja **Automatyzowanie przyjmowania zamówień ML** została stworzona w celu automatyzacji procesu przyjmowania zamówień. Jej główne funkcjonalności to:

- Pobieranie wiadomości e-mail z konta firmowego.
- Przetwarzanie treści wiadomości oraz załączników (np. PDF) na czytelny format tekstowy.
- Wyciąganie kluczowych informacji z wiadomości przy użyciu algorytmów uczenia maszynowego (ML).

Obecnie aplikacja obsługuje część odpowiedzialną za pobieranie i przetwarzanie wiadomości e-mail. Instrukcje dotyczące konfiguracji algorytmów ML zostaną dodane w późniejszym czasie.

---

## Wymagania systemowe

- **Python** 3.8 lub nowszy
- Zainstalowane zewnętrzne narzędzia:
  - **Tesseract OCR** (do przetwarzania obrazów w PDF-ach)
  - **Poppler** (do konwersji PDF na obrazy)

---

## Instalacja

1. **Sklonuj repozytorium**:
   ```sh
   git clone https://github.com/TwojaNazwaRepozytorium/AutomatyzowanieZamowienML.git
   cd AutomatyzowanieZamowienML
   ```
2. **Utwórz wirtualne środowisko i aktywuj je:**
    ```sh
    python -m venv venv
    venv\Scripts\activate
    ```
3. **Zainstaluj wymagane biblioteki:**
    ```sh
    pip install -r requirements.txt
    ```
4. **Skonfiguruj ścieżki do Tesseract OCR i Poppler:**
    ```sh
    PYTESSERACT_PATH = r'Ścieżka\do\tesseract.exe'
    POPPLER_PATH = r'Ścieżka\do\poppler\bin'
    ```
5. **Skonfiguruj dane logowania do e-maila:**

## Uruchamianie aplikacji
1. **Uruchom aplikację:**
    ```sh
    python main.py
    ```
2. Aplikacja będzie działać w trybie ciągłym, sprawdzając nowe wiadomości e-mail co 60 sekund. Przetworzone wiadomości zostaną zapisane w folderze processed_emails.

## Funkcjonalności
1. Pobieranie wiadomości e-mail z konta firmowego.
2. Przetwarzanie załączników PDF (OCR dla skanowanych dokumentów).
3. Zapisywanie treści wiadomości w formacie .txt.
## Przyszłe kroki
1. Implementacja algorytmów ML do wyciągania kluczowych informacji z przetworzonych wiadomości.
2. Dodanie szczegółowych instrukcji konfiguracji dla części ML.
