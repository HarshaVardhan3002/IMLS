import os
from pypdf import PdfReader

def extract_text_from_pdfs(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    for file in files:
        path = os.path.join(directory, file)
        print(f"--- CONTENT OF {file} ---")
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                print(page.extract_text())
        except Exception as e:
            print(f"Error reading {file}: {e}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    pdf_dir = r"F:\old_desktop\Desktop (1)\Lecturers SoSe 2026\Introduction to ML Safety\Excercise\2026\2026\all excercises"
    extract_text_from_pdfs(pdf_dir)
