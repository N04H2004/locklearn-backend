# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import openai
import fitz  # pip install pymupdf
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# get API key from env var
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")
openai.api_key = OPENAI_KEY

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)

@app.post("/generate")
async def generate_questions(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = extract_text_from_pdf(pdf_bytes)

    # Prompt: ask model to return pure JSON
    prompt = f"""
Generate exactly 5 multiple-choice questions (MCQ) based on the course text below.
Return only valid JSON (no extra commentary). Format:

{{"questions":[
  {{"question":"...","choices":["A","B","C","D"],"answer_index":0}},
  ...
]}}

Course text:
{text}
"""

    # call OpenAI
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",          # or another available model
        messages=[{"role":"user","content":prompt}],
        max_tokens=1000,
        temperature=0.2
    )

    raw = resp.choices[0].message["content"].strip()

    # try to parse JSON, otherwise try to extract substring that is JSON
    try:
        data = json.loads(raw)
    except Exception:
        # attempt to find first { and last } and parse
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(raw[start:end+1])
            except Exception:
                data = {"error": "Could not parse model output as JSON", "raw": raw}
        else:
            data = {"error": "No JSON found in model output", "raw": raw}

    return data
