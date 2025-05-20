import imaplib, email, os, time, re, json
from email.header import decode_header
from bs4 import BeautifulSoup
from pdf_processor import process_pdf_attachment

SAVED_PATHS_FILE = "e:\\ProjektyGITHUB\\Grupa\\Projekt-Firma\\Data\\SavedPaths.json"

def load_saved_paths():
    """Load saved paths and email configuration from the JSON file."""
    if os.path.exists(SAVED_PATHS_FILE):
        with open(SAVED_PATHS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_saved_paths(data):
    """Save paths and email configuration to the JSON file."""
    with open(SAVED_PATHS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_email_config():
    """Ensure email configuration is set up in the JSON file."""
    data = load_saved_paths()

    if "Email" not in data:
        data["Email"] = {}

    email_config = data["Email"]

    # Check if email configuration exists
    if email_config:
        print(f"Current email configuration:")
        print(f"Provider: {email_config.get('provider', 'Not set')}")
        print(f"Email: {email_config.get('email', 'Not set')}")
        use_existing = input("Do you want to use this email configuration? (yes/no): ").strip().lower()
        while use_existing not in ["yes", "no"]:
            if use_existing == "yes":
                return email_config
            elif use_existing == "no":
                break
            else:
                use_existing = input("Invalid input. Please enter 'yes' or 'no': ").strip().lower()

    # Prompt user for new email configuration
    email_config["provider"] = input("Enter email provider (e.g., gmail, outlook, yahoo, custom): ").strip()
    email_config["email"] = input("Enter your email address: ").strip()
    email_config["password"] = input("Enter your email password: ").strip()

    if email_config["provider"] == "custom":
        email_config["imap_server"] = input("Enter the IMAP server address: ").strip()
    else:
        # Set default IMAP servers for known providers
        if email_config["provider"] == "gmail":
            email_config["imap_server"] = "imap.gmail.com"
        elif email_config["provider"] == "outlook":
            email_config["imap_server"] = "outlook.office365.com"
        elif email_config["provider"] == "yahoo":
            email_config["imap_server"] = "imap.mail.yahoo.com"

    # Save the updated configuration
    data["Email"] = email_config
    save_saved_paths(data)
    return email_config


class EmailExtractor:
    def __init__(self, config):
        self.config = config
        self.connection = None
        self.attachments_dir = "attachments"
        self.processed_dir = "processed_emails"
        
        # Create directories if they don't exist
        os.makedirs(self.attachments_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def connect(self):
        """Connect to the email server"""
        try:
            mail.login(self.config['email'], self.config['password'])
            print("Login successful")
            mail.select('inbox')
            self.connection = mail
            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return False

    def process_emails(self):
        """Process all unread emails"""
        if not self.connection:
            if not self.connect():
                return

        try:
            # Search for all unread messages
            status, messages = self.connection.search(None, '(UNSEEN)')
            if status != 'OK':
                print("No unread emails found")
                return

            email_ids = messages[0].split()
            print(f"Found {len(email_ids)} unread emails")

            for email_id in email_ids:
                self.process_single_email(email_id)

        except Exception as e:
            print(f"Error processing emails: {str(e)}")
            # Try to reconnect on error
            self.connect()

    def process_single_email(self, email_id):
        """Process a single email message"""
        try:
            # Fetch the email
            status, msg_data = self.connection.fetch(email_id, '(RFC822)')
            if status != 'OK':
                print(f"Failed to fetch email {email_id}")
                return

            raw_email = msg_data[0][1]
            email_message = email.message_from_bytes(raw_email)

            # Get email details
            subject, encoding = decode_header(email_message['Subject'])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or 'utf-8')
            from_ = email.utils.parseaddr(email_message.get('From'))[1]

            print(f"\nProcessing email from {from_} with subject: {subject}")

            # Extract content and attachments
            forwarded_content = self.extract_content(email_message)
            self.save_attachments(email_message, email_id)

            if forwarded_content:
                # Save the forwarded content
                self.save_email_content(subject, from_, forwarded_content, email_id)

                # Mark as read
                self.connection.store(email_id, '+FLAGS', '\\Seen')
                print(f"Marked email {email_id} as read")

        except Exception as e:
            print(f"Error processing email {email_id}: {str(e)}")

    def extract_content(self, email_message):
        """Extract the forwarded content from email, including PDF attachments"""
        body = ""
        pdf_texts = []

        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                # Skip non-text attachments
                if "attachment" in content_disposition:
                    # Process PDF attachments
                    print(f"Processing attachment: {content_disposition}")
                    pdf_text = process_pdf_attachment(part, email_message['Message-ID'], self.attachments_dir)
                    if pdf_text:
                        pdf_texts.append(pdf_text)
                    continue

                # Get text content
                if content_type == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode()
                    except:
                        body += str(part.get_payload(decode=True))
                elif content_type == "text/html":
                    try:
                        html = part.get_payload(decode=True).decode()
                        soup = BeautifulSoup(html, 'html.parser')
                        body += soup.get_text()
                    except:
                        pass
        else:
            try:
                body = email_message.get_payload(decode=True).decode()
            except:
                body = str(email_message.get_payload(decode=True))

        # Append extracted PDF text to the email body
        if pdf_texts:
            body += "\n\n--- Extracted PDF Content ---\n"
            body += "\n\n".join(pdf_texts)

        # Try to extract forwarded content
        forwarded_content = self.extract_forwarded_content(body)
        return forwarded_content if forwarded_content else body

    def extract_forwarded_content(self, body):
        """Try to identify and extract forwarded content"""
        patterns = [
            r'-----Original Message-----.*?(.*)',
            r'Begin forwarded message:.*?(.*)',
            r'From:.*?(.*)',
            r'Date:.*?(.*)',
            r'Subject:.*?(.*)',
            r'To:.*?(.*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def save_attachments(self, email_message, email_id):
        """Save all attachments from the email"""
        for part in email_message.walk():
            content_disposition = str(part.get("Content-Disposition"))
            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    # Decode filename if needed
                    filename_header = decode_header(filename)[0]
                    if isinstance(filename_header[0], bytes):
                        filename = filename_header[0].decode(filename_header[1] or 'utf-8')
                    
                    filepath = os.path.join(self.attachments_dir, f"{email_id}_{filename}")
                    with open(filepath, 'wb') as f:
                        f.write(part.get_payload(decode=True))
                    print(f"Saved attachment: {filename}")

    def save_email_content(self, subject, from_, content, email_id):
        """Save the email content to a file"""
        # Clean the subject for filename
        clean_subject = re.sub(r'[^\w\s-]', '', subject).strip()
        clean_subject = re.sub(r'[-\s]+', '_', clean_subject)
        
        filename = f"{email_id}_{from_}_{clean_subject[:50]}.txt"
        filepath = os.path.join(self.processed_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"From: {from_}\n")
            f.write(f"Subject: {subject}\n\n")
            f.write(content)
        
        print(f"Saved email content to {filename}")

    def run_continuously(self, interval=60):
        """Run the email processing continuously"""
        print("Starting email monitoring service...")
        while True:
            try:
                self.process_emails()
                print(f"Waiting {interval} seconds before next check...")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nStopping email monitoring...")
                if self.connection:
                    self.connection.close()
                break
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(interval * 2)  # Wait longer after error
                self.connect()  # Try to reconnect

if __name__ == '__main__':
    email_config = get_email_config()
    extractor = EmailExtractor(email_config)
    extractor.run_continuously(interval=60)