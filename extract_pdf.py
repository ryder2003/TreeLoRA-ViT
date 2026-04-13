import PyPDF2
import sys

pdf_path = "2506.10355v1 (1).pdf"
output_path = "paper_content.txt"

try:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        text_content = []
        print(f"Total pages: {len(pdf_reader.pages)}")
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            text_content.append(f"\n{'='*80}\nPAGE {page_num + 1}\n{'='*80}\n{text}")
        
        full_text = "\n".join(text_content)
        
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(full_text)
        
        print(f"Successfully extracted text to {output_path}")
        print(f"Total characters extracted: {len(full_text)}")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
