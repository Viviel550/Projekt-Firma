import logging
import imaplib
import email
from email.header import decode_header
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class EmailProcessor:
    """Handles email processing and enhancement"""
    
    def enhance_extraction(self, data: Dict[str, Any], email_content: str) -> Dict[str, Any]:
        """Enhance email extraction with email-specific patterns"""
        # Add email-specific enhancements
        if 'bestelling' in email_content.lower() or 'order' in email_content.lower():
            data['email_type'] = 'order'
        elif 'levering' in email_content.lower():
            data['email_type'] = 'delivery'
        elif 'factuur' in email_content.lower():
            data['email_type'] = 'invoice'
        else:
            data['email_type'] = 'unknown'
        
        return data
    
    def read_from_imap(self, server: str, username: str, password: str, 
                       folder: str, extractor, unread_only: bool = True, 
                       mark_as_read: bool = True) -> List[Dict[str, Any]]:
        """Read emails from IMAP server"""
        try:
            # Connect to IMAP server
            logger.info(f"Connecting to IMAP server: {server}")
            mail = imaplib.IMAP4_SSL(server)
            mail.login(username, password)
            mail.select(folder)
            
            # Search for emails
            if unread_only:
                search_criteria = "UNSEEN"  # Only unread emails
                logger.info("Searching for unread emails...")
            else:
                search_criteria = "ALL"
                logger.info("Searching for all emails...")
            
            status, messages = mail.search(None, search_criteria)
            
            if status != 'OK':
                logger.error("Failed to search emails")
                return []
            
            email_ids = messages[0].split()
            logger.info(f"Found {len(email_ids)} emails to process")
            
            if not email_ids:
                logger.info("No emails found to process")
                mail.logout()
                return []
            
            processed_emails = []
            
            for email_id in email_ids:
                try:
                    # Fetch email
                    status, msg_data = mail.fetch(email_id, '(RFC822)')
                    
                    if status != 'OK':
                        logger.warning(f"Failed to fetch email ID: {email_id}")
                        continue
                    
                    # Parse email
                    email_body = msg_data[0][1]
                    email_message = email.message_from_bytes(email_body)
                    
                    # Extract email content
                    email_content = self._extract_email_content(email_message)
                    
                    if not email_content.strip():
                        logger.warning(f"Empty email content for ID: {email_id}")
                        continue
                    
                    # Process email with extractor
                    extracted_data = extractor.extract_from_email(email_content)
                    
                    # Add email metadata
                    extracted_data.update({
                        'email_id': email_id.decode(),
                        'email_subject': self._decode_email_header(email_message.get('Subject', '')),
                        'email_from': email_message.get('From', ''),
                        'email_date': email_message.get('Date', ''),
                        'source_type': 'email'
                    })
                    
                    processed_emails.append(extracted_data)
                    
                    # Mark as read if requested
                    if mark_as_read and unread_only:
                        mail.store(email_id, '+FLAGS', '\\Seen')
                        logger.debug(f"Marked email ID {email_id} as read")
                    
                    logger.info(f"Successfully processed email ID: {email_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing email ID {email_id}: {str(e)}")
                    continue
            
            # Close connection
            mail.close()
            mail.logout()
            
            logger.info(f"Successfully processed {len(processed_emails)} emails")
            return processed_emails
            
        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error reading emails from IMAP: {str(e)}")
            return []
    
    def _extract_email_content(self, email_message) -> str:
        """Extract text content from email message"""
        content = ""
        
        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    # Extract text content
                    if content_type == "text/plain":
                        body = part.get_payload(decode=True)
                        if body:
                            content += body.decode('utf-8', errors='ignore') + "\n"
                    elif content_type == "text/html":
                        # Basic HTML to text conversion
                        body = part.get_payload(decode=True)
                        if body:
                            html_content = body.decode('utf-8', errors='ignore')
                            # Simple HTML tag removal
                            import re
                            text_content = re.sub('<[^<]+?>', '', html_content)
                            content += text_content + "\n"
            else:
                # Single part email
                content_type = email_message.get_content_type()
                if content_type in ["text/plain", "text/html"]:
                    body = email_message.get_payload(decode=True)
                    if body:
                        if content_type == "text/html":
                            import re
                            html_content = body.decode('utf-8', errors='ignore')
                            content = re.sub('<[^<]+?>', '', html_content)
                        else:
                            content = body.decode('utf-8', errors='ignore')
        
        except Exception as e:
            logger.error(f"Error extracting email content: {str(e)}")
        
        return content.strip()
    
    def _decode_email_header(self, header: str) -> str:
        """Decode email header"""
        try:
            decoded_header = decode_header(header)
            decoded_string = ""
            
            for part, encoding in decoded_header:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_string += part.decode(encoding)
                    else:
                        decoded_string += part.decode('utf-8', errors='ignore')
                else:
                    decoded_string += part
            
            return decoded_string
        except Exception as e:
            logger.error(f"Error decoding email header: {str(e)}")
            return header