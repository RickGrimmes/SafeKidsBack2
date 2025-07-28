from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
import requests
from io import BytesIO

app = FastAPI()

@app.get("/")
def read_root():
    return JSONResponse(content={"mensaje": "Hola mundo"})

@app.get("/imagechingona")
def image_chingona():
    url = "https://static.wikia.nocookie.net/heroe/images/c/c4/AiAi_SMBBR.png/revision/latest?cb=20240710005028&path-prefix=es"
    response = requests.get(url)
    if response.status_code == 200:
        return StreamingResponse(BytesIO(response.content), media_type="image/png")
    return JSONResponse(content={"error": "No se pudo obtener la imagen"}, status_code=404)
