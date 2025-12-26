import io
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def calculate_match(resume_text, job_desc):
    content = [resume_text, job_desc]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(content)
    
    # Using Cosine Similarity to find the match percentage
    similarity_matrix = cosine_similarity(count_matrix)
    match_percentage = similarity_matrix[0][1] * 100
    return round(match_percentage, 2)

# --- EXECUTION ---
# Copy the Job Description text from the LinkedIn screenshot
job_description = """
Software Engineering Intern. Skills: React, Python, Node.js, 
Git, HTML, CSS, Javascript. AI curiosity, proactive, remote discipline.
"""

# Put your resume.pdf in the same folder
resume_path = "your_resume.pdf" 

try:
    resume_text = extract_text_from_pdf(resume_path)
    score = calculate_match(resume_text, job_description)
    print(f"--- Screening Result ---")
    print(f"Match Score: {score}%")
except Exception as e:
    print(f"Error: {e}. Make sure 'your_resume.pdf' is in the folder.")