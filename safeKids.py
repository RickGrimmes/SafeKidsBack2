from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import requests
from io import BytesIO
from deepface import DeepFace
import os
os.environ["DEEPPFACE_HOME"] = "C:/Users/babaj/.deepface"

app = FastAPI()

# --- Constantes de tipos permitidos ---
TIPOS_PERMITIDOS = {"AUTHORIZEDS", "GUARDIANS", "STUDENTS", "USERS"}

# --- Función auxiliar para guardar imagen ---
def guardar_imagen(escuela, tipo, file: UploadFile):
    if tipo not in TIPOS_PERMITIDOS:
        return False, f"Tipo de usuario no permitido: {tipo}"
    ruta = os.path.join("C:/Users/babaj/Documents/9C/IMAGES", escuela, tipo)
    os.makedirs(ruta, exist_ok=True)
    nombre_archivo = file.filename
    ruta_completa = os.path.join(ruta, nombre_archivo)
    with open(ruta_completa, "wb") as f:
        f.write(file.file.read())
    return True, ruta_completa

# --- Endpoints para cada tipo de usuario ---

#region AUTHORIZED PEOPLE

# --- Endpoint para subir imagen de authorized people ---
@app.post("/upload/authorizeds")
async def upload_authorizeds(escuela: str = File(...), file: UploadFile = File(...)):
    ruta_escuela = os.path.join("C:/Users/babaj/Documents/9C/IMAGES", escuela)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"error": f"La escuela '{escuela}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(escuela, "AUTHORIZEDS", file)
    if ok:
        return {"mensaje": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"error": msg}, status_code=400)

# --- Endpoint para buscar el authorized people más parecido en una escuela ---
@app.post("/busca/authorizeds")
async def busca_authorizeds(escuela: str = File(...), file: UploadFile = File(...)):
    import numpy as np
    from PIL import Image
    from deepface import DeepFace
    import glob
    ruta_authorizeds = os.path.join("C:/Users/babaj/Documents/9C/IMAGES", escuela, "AUTHORIZEDS")
    if not os.path.isdir(ruta_authorizeds):
        return JSONResponse(content={"error": f"La carpeta de authorized en la escuela '{escuela}' no existe."}, status_code=400)
    # Leer imagen enviada
    contents = await file.read()
    img_query = Image.open(BytesIO(contents))
    img_query_np = np.array(img_query)
    # Buscar todas las imágenes en la carpeta usando DeepFace.find
    if not os.listdir(ruta_authorizeds):
        return JSONResponse(content={"error": "No hay imágenes en la carpeta authorized."}, status_code=404)
    # Guardar la imagen recibida temporalmente
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img_query.save(tmp, format='JPEG')
        tmp_path = tmp.name
    try:
        df = DeepFace.find(img_path=tmp_path, db_path=ruta_authorizeds, enforce_detection=False, model_name='Facenet')
    finally:
        os.remove(tmp_path)
    # DeepFace.find puede devolver un DataFrame o una lista de DataFrames
    if isinstance(df, list):
        df = df[0]
    if df.empty:
        return JSONResponse(content={"error": "No se encontró ninguna coincidencia válida."}, status_code=404)
    mejor = df.iloc[0]
    porcentaje = round((1 - mejor['distance']) * 100, 2)
    if porcentaje <= 60:
        return JSONResponse(content={"error": "No se encontró ninguna coincidencia con al menos 60% de similitud."}, status_code=404)
    return {"archivo": os.path.basename(mejor['identity']), "porcentaje_similitud": porcentaje}
#endregion

#region GUARDIANS

# --- Endpoint para subir imagen de guardian ---
@app.post("/upload/guardians")
async def upload_guardians(escuela: str = File(...), file: UploadFile = File(...)):
    ruta_escuela = os.path.join("C:/Users/babaj/Documents/9C/IMAGES", escuela)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"error": f"La escuela '{escuela}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(escuela, "GUARDIANS", file)
    if ok:
        return {"mensaje": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"error": msg}, status_code=400)

# --- Endpoint para buscar el guardian más parecido en una escuela ---
@app.post("/busca/guardian")
async def busca_guardian(escuela: str = File(...), file: UploadFile = File(...)):
    import numpy as np
    from PIL import Image
    from deepface import DeepFace
    import glob
    ruta_guardians = os.path.join("C:/Users/babaj/Documents/9C/IMAGES", escuela, "GUARDIANS")
    if not os.path.isdir(ruta_guardians):
        return JSONResponse(content={"error": f"La carpeta de guardians en la escuela '{escuela}' no existe."}, status_code=400)
    # Leer imagen enviada
    contents = await file.read()
    img_query = Image.open(BytesIO(contents))
    img_query_np = np.array(img_query)
    # Buscar todas las imágenes en la carpeta usando DeepFace.find
    if not os.listdir(ruta_guardians):
        return JSONResponse(content={"error": "No hay imágenes en la carpeta guardians."}, status_code=404)
    # Guardar la imagen recibida temporalmente
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img_query.save(tmp, format='JPEG')
        tmp_path = tmp.name
    try:
        df = DeepFace.find(img_path=tmp_path, db_path=ruta_guardians, enforce_detection=False, model_name='Facenet')
    finally:
        os.remove(tmp_path)
    # DeepFace.find puede devolver un DataFrame o una lista de DataFrames
    if isinstance(df, list):
        df = df[0]
    if df.empty:
        return JSONResponse(content={"error": "No se encontró ninguna coincidencia válida."}, status_code=404)
    mejor = df.iloc[0]
    porcentaje = round((1 - mejor['distance']) * 100, 2)
    if porcentaje <= 60:
        return JSONResponse(content={"error": "No se encontró ninguna coincidencia con al menos 60% de similitud."}, status_code=404)
    return {"archivo": os.path.basename(mejor['identity']), "porcentaje_similitud": porcentaje}
#endregion

#region STUDENTS

# --- Endpoint para subir imagen de student ---
@app.post("/upload/students")
async def upload_students(escuela: str = File(...), file: UploadFile = File(...)):
    ruta_escuela = os.path.join("C:/Users/babaj/Documents/9C/IMAGES", escuela)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"error": f"La escuela '{escuela}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(escuela, "STUDENTS", file)
    if ok:
        return {"mensaje": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"error": msg}, status_code=400)

# --- Endpoint para buscar el student más parecido en una escuela ---
@app.post("/busca/student")
async def busca_student(escuela: str = File(...), file: UploadFile = File(...)):
    import numpy as np
    from PIL import Image
    from deepface import DeepFace
    import glob
    ruta_students = os.path.join("C:/Users/babaj/Documents/9C/IMAGES", escuela, "STUDENTS")
    if not os.path.isdir(ruta_students):
        return JSONResponse(content={"error": f"La carpeta de students en la escuela '{escuela}' no existe."}, status_code=400)
    # Leer imagen enviada
    contents = await file.read()
    img_query = Image.open(BytesIO(contents))
    img_query_np = np.array(img_query)
    # Buscar todas las imágenes en la carpeta usando DeepFace.find
    if not os.listdir(ruta_students):
        return JSONResponse(content={"error": "No hay imágenes en la carpeta students."}, status_code=404)
    # Guardar la imagen recibida temporalmente
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img_query.save(tmp, format='JPEG')
        tmp_path = tmp.name
    try:
        df = DeepFace.find(img_path=tmp_path, db_path=ruta_students, enforce_detection=False, model_name='Facenet')
    finally:
        os.remove(tmp_path)
    # DeepFace.find puede devolver un DataFrame o una lista de DataFrames
    if isinstance(df, list):
        df = df[0]
    if df.empty:
        return JSONResponse(content={"error": "No se encontró ninguna coincidencia válida."}, status_code=404)
    mejor = df.iloc[0]
    porcentaje = round((1 - mejor['distance']) * 100, 2)
    if porcentaje <= 60:
        return JSONResponse(content={"error": "No se encontró ninguna coincidencia con al menos 60% de similitud."}, status_code=404)
    return {"archivo": os.path.basename(mejor['identity']), "porcentaje_similitud": porcentaje}
#endregion

#region USERS

# --- Endpoint para subir imagen de user ---
@app.post("/upload/users")
async def upload_users(escuela: str = File(...), file: UploadFile = File(...)):
    ruta_escuela = os.path.join("C:/Users/babaj/Documents/9C/IMAGES", escuela)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"error": f"La escuela '{escuela}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(escuela, "USERS", file)
    if ok:
        return {"mensaje": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"error": msg}, status_code=400)

# --- Endpoint para buscar el user más parecido en una escuela ---
@app.post("/busca/user")
async def busca_user(escuela: str = File(...), file: UploadFile = File(...)):
    import numpy as np
    from PIL import Image
    from deepface import DeepFace
    import glob
    ruta_users = os.path.join("C:/Users/babaj/Documents/9C/IMAGES", escuela, "USERS")
    if not os.path.isdir(ruta_users):
        return JSONResponse(content={"error": f"La carpeta de users en la escuela '{escuela}' no existe."}, status_code=400)
    # Leer imagen enviada
    contents = await file.read()
    img_query = Image.open(BytesIO(contents))
    img_query_np = np.array(img_query)
    # Buscar todas las imágenes en la carpeta usando DeepFace.find
    if not os.listdir(ruta_users):
        return JSONResponse(content={"error": "No hay imágenes en la carpeta users."}, status_code=404)
    # Guardar la imagen recibida temporalmente
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img_query.save(tmp, format='JPEG')
        tmp_path = tmp.name
    try:
        df = DeepFace.find(img_path=tmp_path, db_path=ruta_users, enforce_detection=False, model_name='Facenet')
    finally:
        os.remove(tmp_path)
    # DeepFace.find puede devolver un DataFrame o una lista de DataFrames
    if isinstance(df, list):
        df = df[0]
    if df.empty:
        return JSONResponse(content={"error": "No se encontró ninguna coincidencia válida."}, status_code=404)
    mejor = df.iloc[0]
    porcentaje = round((1 - mejor['distance']) * 100, 2)
    if porcentaje <= 60:
        return JSONResponse(content={"error": "No se encontró ninguna coincidencia con al menos 60% de similitud."}, status_code=404)
    return {"archivo": os.path.basename(mejor['identity']), "porcentaje_similitud": porcentaje}
#endregion

#region PRUEBA

@app.get("/imagendurisima")
def image_chingona():
    url = "https://static.wikia.nocookie.net/heroe/images/c/c4/AiAi_SMBBR.png/revision/latest?cb=20240710005028&path-prefix=es"
    response = requests.get(url)
    if response.status_code == 200:
        return StreamingResponse(BytesIO(response.content), media_type="image/png")
    return JSONResponse(content={"error": "No se pudo obtener la imagen"}, status_code=404)

# Nuevo endpoint para detectar si hay una persona en la imagen
@app.post("/detecta")
async def detecta_persona(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_bytes = BytesIO(contents)
        import numpy as np
        from PIL import Image
        img = Image.open(img_bytes)
        img_np = np.array(img)
        DeepFace.analyze(img_np, actions=["emotion"], enforce_detection=True)
        return {"persona": True, "mensaje": "Sí hay una persona en la imagen"}
    except Exception:
        return {"persona": False, "mensaje": "No se detectó ningún rostro"}
#endregion