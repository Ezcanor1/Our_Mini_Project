import pdfplumber

def extract_text_pdf(filepath):
    """
    Extracts text from a PDF file.
    """
    try:
        with pdfplumber.open(filepath) as pdf:
            pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
        return '\n'.join(pages)
    except Exception as e:
        print(f"Error processing PDF file: {e}")
        return ""
