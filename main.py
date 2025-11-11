from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import openai
import fitz  # PyMuPDF pour lire le PDF

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = "TON_API_KEY_OPENAI"

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.post("/generate")
async def generate_questions(file: UploadFile = File(...)):
    pdf_content = await file.read()
    text = extract_text_from_pdf(pdf_content)

    prompt = f"""
    Generate 5 multiple choice questions based on the course below.
    Format:
    [
      {{"question":"...", "choices":["A","B","C","D"], "answer":"A"}}
    ]

    Course content:
    {text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message["content"]
