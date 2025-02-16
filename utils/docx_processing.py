from docx import Document

def extract_text_docx(filepath):
    """
    Extracts text from a DOCX file.
    """
    try:
        doc = Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error processing DOCX file: {e}")
        return ""
