from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Autorise les requêtes depuis ton frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👉 Serve le fichier static/index.html sur route /
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


# 👉 Endpoint pour upload fichier et générer questions
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8")

    prompt = f"""
    Génère 5 questions d'examen sur le contenu suivant en français.
    Format en JSON :
    {{
      "questions": [
        "question 1",
        "question 2",
        "question 3",
        "question 4",
        "question 5"
      ]
    }}

    --- CONTENU DU DOCUMENT ---
    {content}
    """

    response = client.chat.completions.create(
        model="gpt-5",  # tu peux mettre gpt-4 si besoin
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return {"questions": response.choices[0].message["content"]}


# Mode local (sur Render ils ignorent cette partie, c’est OK)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
