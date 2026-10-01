"""Extract bounded report text without treating uploaded content as instructions."""
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import re

MAX_BYTES = 10 * 1024 * 1024
MAX_CHARS = 18000
MAX_FILES = 3


def is_account_file(name):
    """Keep account-recovery material out of medical report processing."""
    return bool(re.search(r'(?:recovery[ _-]*codes?|backup[ _-]*codes?|credentials|passwords?)', Path(name).name, re.I))


def read_report(name, content):
    if is_account_file(name):
        raise ValueError('This looks like an account or recovery-code file. Please choose a medical report, test result or prescription instead.')
    if not content or len(content) > MAX_BYTES:
        raise ValueError('Each file must be nonempty and no larger than 10 MB.')
    extension = Path(name).suffix.lower()
    if extension == '.pdf':
        from pypdf import PdfReader
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise ValueError('Upload an unlocked PDF.')
            if len(reader.pages) > 30:
                raise ValueError('Please upload a report with no more than 30 pages.')
            parts = []
            for index, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ''
                if not text.strip():
                    raise ValueError(f'Page {index} has no readable text. Upload a clear photo of that page instead.')
                parts.append(f'[Page {index}]\n{text}')
            text = '\n\n'.join(parts)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError('This PDF could not be read. Upload an unlocked, text-based PDF.') from error
    elif extension == '.txt':
        try:
            text = '[Text file]\n' + content.decode('utf-8-sig')
        except UnicodeDecodeError as error:
            raise ValueError('Please save the text file as UTF-8.') from error
    elif extension in {'.png', '.jpg', '.jpeg'}:
        text = '[Image OCR; verify numbers against the original]\n' + image_text(content, extension)
    else:
        raise ValueError('Supported files: PDF, TXT, PNG and JPEG.')
    if not text.strip() or not any(c.isalnum() for c in text.split('\n', 1)[-1]):
        raise ValueError('No readable text was found. Please upload a clearer file.')
    if len(text) > MAX_CHARS:
        raise ValueError('This report is too long. Upload the relevant pages separately.')
    return {'name': Path(name).name[:160], 'text': text}


def image_text(content, extension):
    from PIL import Image
    try:
        with Image.open(BytesIO(content)) as image:
            if image.width * image.height > 25000000:
                raise ValueError('Please resize the photo to less than 25 megapixels.')
            image.verify()
    except Exception as error:
        raise ValueError('This image could not be read. Please upload a clear PNG or JPEG.') from error
    with tempfile.TemporaryDirectory(prefix='mrs-report-') as directory:
        path = Path(directory) / ('report' + extension)
        path.write_bytes(content)
        if sys.platform == 'darwin':
            command = ['/usr/bin/swift', '-module-cache-path', directory + '/cache',
                       str(Path(__file__).with_name('report_ocr.swift')), str(path)]
        elif shutil.which('tesseract'):
            command = ['tesseract', str(path), 'stdout']
        else:
            raise ValueError('Photo reading is unavailable on this server. Upload a text-based PDF or TXT file.')
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError('Photo reading did not finish. Try a text-based PDF.') from error
        if result.returncode or not result.stdout.strip():
            raise ValueError('No readable text found in this photo. Try a clearer image.')
        return result.stdout.strip()
