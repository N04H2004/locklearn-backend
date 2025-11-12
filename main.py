# main.py
import os
import io
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# PyPDF2 pour extraire le texte depuis les PDF
from PyPDF2 import PdfReader

# OpenAI SDK (utiliser la lib que tu as installée)
# Ici on utilise l'interface "OpenAI" moderne ; si tu utilises une autre lib adapte.
from openai import OpenAI

# Initialise le client OpenAI avec la variable d'environnement OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. OpenAI calls will fail if not configured.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="LockLearn - PDF → 5 Questions (proto)")

# Autoriser toutes origines pour prototypage (en prod, restreindre)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from ./static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    """
    Sert le fichier static/index.html à la racine.
    """
    index_path = "static/index.html"
    if not os.path.exists(index_path):
        return JSONResponse(status_code=404, content={"error": "Frontend not found. Add static/index.html"})
    return FileResponse(index_path)


def extract_text_from_pdf_bytes(b: bytes) -> str:
    """
    Extrait le texte d'un PDF (bytes) via PyPDF2.
    Retourne la chaîne concaténée des pages (vide si échec).
    """
    try:
        stream = io.BytesIO(b)
        reader = PdfReader(stream)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        # log server-side pour debug
        print("PDF extraction error:", e)
        return ""


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint pour uploader un fichier (PDF ou TXT).
    - Extrait le texte (PDF via PyPDF2, .txt via decode utf-8)
    - Construit un prompt
    - Appelle OpenAI pour générer 5 questions en français
    - Renvoie un JSON structuré : { "questions": [ ... ] } ou { "questions_text": "..." }
    """
    try:
        if not file:
            return JSONResponse(status_code=400, content={"error": "No file uploaded"})

        filename = file.filename or "uploaded_file"
        content_type = (file.content_type or "").lower()

        raw = await file.read()

        # === Extraction du texte ===
        text = ""
        if filename.lower().endswith(".pdf") or "pdf" in content_type:
            text = extract_text_from_pdf_bytes(raw)
            if not text:
                return JSONResponse(status_code=400, content={"error": "Unable to extract text from PDF"})
        else:
            # try utf-8 decode for plain text files
            try:
                text = raw.decode("utf-8")
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": "Unsupported file format or encoding", "detail": str(e)})

        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "No text extracted from file"})

        # safe length (pour éviter d'envoyer un prompt monstrueux)
        text = text.strip().replace("\n", " ")
        if len(text) > 25000:
            text = text[:25000]

        # === Prompt to OpenAI ===
        prompt = f"""
Génère 5 questions d'examen en français à partir du texte suivant.
Retourne strictement un JSON valide au format :
{{ "questions": ["question 1", "question 2", "question 3", "question 4", "question 5"] }}

Texte:
{text}
"""

        # === Appel OpenAI ===
        try:
            # Ajuste le modèle selon ton accès (gpt-4, gpt-4o-mini, gpt-5, etc.)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600
            )
            # récupère le contenu textuel
            content = response.choices[0].message["content"]
        except Exception as e:
            print("OpenAI call failed:", e)
            return JSONResponse(status_code=502, content={"error": "OpenAI request failed", "detail": str(e)})

        # === Tenter de parser la réponse modèle en JSON ===
        try:
            parsed = json.loads(content)
            # si c'est au bon format, renvoyer directement
            if isinstance(parsed, dict) and "questions" in parsed:
                return JSONResponse(status_code=200, content=parsed)
            else:
                # Sinon fallback : renvoyer le texte brut sous questions_text
                return JSONResponse(status_code=200, content={"questions_text": content})
        except Exception:
            # modèle n'a pas renvoyé du JSON — fallback
            return JSONResponse(status_code=200, content={"questions_text": content})

    except HTTPException:
        raise
    except Exception as e:
        print("Upload route unexpected error:", e)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": str(e)})


# Optionnel : endpoint health-check
@app.get("/health")
def health():
    return {"status": "ok"}


# Exécution en local (Render ignore souvent cette partie mais OK pour dev local)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
