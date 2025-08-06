
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
import requests
from io import BytesIO
from deepface import DeepFace
import os
os.environ["DEEPPFACE_HOME"] = "C:/Users/babaj/.deepface"

# Variable global para la ruta de imágenes
IMG_ROUTE = "C:/Users/babaj/Documents/9C/IMAGES"
# IMG_ROUTE = "/home/carlos/img"

app = FastAPI()

# --- Constantes de tipos permitidos ---
TIPOS_PERMITIDOS = {"AUTHORIZEDS", "GUARDIANS", "STUDENTS", "USERS"}

#region UploadImages

# --- Función auxiliar para guardar imagen ---
import unicodedata
def quitar_acentos(texto):
    return ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))

def guardar_imagen(escuela, tipo, file: UploadFile, id: str, firstName: str, lastName: str):
    if tipo not in TIPOS_PERMITIDOS:
        return False, f"Tipo de usuario no permitido: {tipo}"
    from PIL import Image
    ruta = os.path.join(IMG_ROUTE, escuela, tipo)
    os.makedirs(ruta, exist_ok=True)
    # Construir nombre de archivo: ID_FIRSTNAMELASTNAME.jpg, todo mayúsculas y sin acentos
    id_clean = quitar_acentos(str(id)).upper()
    first_clean = quitar_acentos(str(firstName)).replace(' ', '').upper()
    last_clean = quitar_acentos(str(lastName)).replace(' ', '').upper()
    nombre_archivo = f"{id_clean}_{first_clean}{last_clean}.jpg"
    ruta_completa = os.path.join(ruta, nombre_archivo)
    try:
        # Validar si ya existe una imagen con ese nombre
        if os.path.exists(ruta_completa):
            return False, "Ya existe una imagen de ese usuario. No se puede sobrescribir." 
        contents = file.file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        # Validar que haya exactamente un rostro
        import numpy as np
        from deepface import DeepFace
        import cv2
        img_np = np.array(img)
        try:
            deteccion = DeepFace.extract_faces(img_np, enforce_detection=True)
        except Exception as e:
            return False, "No se detectó ningún rostro en la imagen. Por favor, sube una foto donde se vea claramente tu cara."
        if not isinstance(deteccion, list) or len(deteccion) != 1:
            return False, "La imagen debe contener exactamente un rostro visible y bien definido."
        # Validación de tamaño mínimo del rostro
        face = deteccion[0]
        region = face.get('facial_area') or face.get('region')
        if not region:
            return False, "No se pudo determinar el área del rostro."
        w = region.get('w') or region.get('width')
        h = region.get('h') or region.get('height')
        if not w or not h:
            return False, "No se pudo determinar el tamaño del rostro."
        min_side = min(img_np.shape[0], img_np.shape[1])
        if w < min_side * 0.2 or h < min_side * 0.2:
            return False, "El rostro es demasiado pequeño en la imagen. Acércate más a la cámara."
        # Validación de nitidez (enfoque)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        if laplacian_var < 30:
            return False, "La imagen está borrosa. Por favor, asegúrate de que la foto esté bien enfocada."
        # Validación de brillo
        brightness = np.mean(img_gray)
        if brightness < 50:
            return False, "La imagen está demasiado oscura. Busca mejor iluminación."
        if brightness > 230:
            return False, "La imagen está demasiado clara o sobreexpuesta."
        # Si pasa todas las validaciones, guardar como JPEG comprimido
        img.save(ruta_completa, format="JPEG", quality=80, optimize=True)
        return True, ruta_completa
    except Exception as e:
        return False, f"Error al guardar la imagen: {e}"

# --- Endpoint para subir imagen de authorized people ---
@app.post("/api2/upload/authorizeds")
async def upload_authorizeds(school_id: int = File(...), id: str = File(...), firstName: str = File(...), lastName: str = File(...), file: UploadFile = File(...)):
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"success": False, "message": f"La escuela '{school_id}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(nombre_carpeta, "AUTHORIZEDS", file, id, firstName, lastName)
    if ok:
        return {"success": True, "message": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"success": False, "message": msg}, status_code=400)

# --- Endpoint para subir imagen de guardian ---
@app.post("/api2/upload/guardians")
async def upload_guardians(school_id: int = File(...), id: str = File(...), firstName: str = File(...), lastName: str = File(...), file: UploadFile = File(...)):
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"success": False, "message": f"La escuela '{school_id}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(nombre_carpeta, "GUARDIANS", file, id, firstName, lastName)
    if ok:
        return {"success": True, "message": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"success": False, "message": msg}, status_code=400)




# --- Endpoint para subir imagen de user ---
@app.post("/api2/upload/users")
async def upload_users(school_id: int = File(...), id: str = File(...), firstName: str = File(...), lastName: str = File(...), file: UploadFile = File(...)):
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"success": False, "message": f"La escuela '{school_id}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(nombre_carpeta, "USERS", file, id, firstName, lastName)
    if ok:
        return {"success": True, "message": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"success": False, "message": msg}, status_code=400)

#endregion  

#region BuscaPersonas

# --- Endpoint para buscar el guardian más parecido en una escuela, si no, entonces busca al authorized ---
@app.post("/api2/busca/guardianAuthPeople")
async def busca_guardian(escuela: str = File(...), file: UploadFile = File(...)):
    import numpy as np
    from PIL import Image
    from deepface import DeepFace
    import glob
    import tempfile
    # Leer imagen enviada
    contents = await file.read()
    img_query = Image.open(BytesIO(contents))
    img_query_np = np.array(img_query)
    # --- Buscar en GUARDIANS ---
    ruta_guardians = os.path.join(IMG_ROUTE, escuela, "GUARDIANS")
    if os.path.isdir(ruta_guardians) and os.listdir(ruta_guardians):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img_query.save(tmp, format='JPEG')
            tmp_path = tmp.name
        try:
            df = DeepFace.find(img_path=tmp_path, db_path=ruta_guardians, enforce_detection=False, model_name='Facenet')
        finally:
            os.remove(tmp_path)
        if isinstance(df, list):
            df = df[0]
        if not df.empty:
            for _, row in df.iterrows():
                porcentaje = round((1 - row['distance']) * 100, 2)
                if porcentaje >= 80:
                    return {"success": True, "message": "Coincidencia encontrada en GUARDIANS", "data": {"archivo": os.path.basename(row['identity']), "porcentaje_similitud": porcentaje, "tipo": "GUARDIAN"}}
    # --- Buscar en AUTHORIZEDS ---
    ruta_authorizeds = os.path.join(IMG_ROUTE, escuela, "AUTHORIZEDS")
    if os.path.isdir(ruta_authorizeds) and os.listdir(ruta_authorizeds):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img_query.save(tmp, format='JPEG')
            tmp_path = tmp.name
        try:
            df = DeepFace.find(img_path=tmp_path, db_path=ruta_authorizeds, enforce_detection=False, model_name='Facenet')
        finally:
            os.remove(tmp_path)
        if isinstance(df, list):
            df = df[0]
        if not df.empty:
            for _, row in df.iterrows():
                porcentaje = round((1 - row['distance']) * 100, 2)
                if porcentaje >= 80:
                    return {"success": True, "message": "Coincidencia encontrada en AUTHORIZEDS", "data": {"archivo": os.path.basename(row['identity']), "porcentaje_similitud": porcentaje, "tipo": "AUTHORIZED"}}
    # Si no encontró nada en ninguno
    return JSONResponse(content={"success": False, "message": "No se encontró ninguna coincidencia válida en guardians ni authorizeds con al menos 80% de similitud."}, status_code=404)

# --- Endpoint para buscar el student más parecido en una escuela OSEA ENTRADA ---
@app.post("/api2/busca/student")
async def busca_student(escuela: str = File(...), file: UploadFile = File(...)):
    import numpy as np
    from PIL import Image
    from deepface import DeepFace
    import glob
    ruta_students = os.path.join(IMG_ROUTE, escuela, "STUDENTS")
    if not os.path.isdir(ruta_students):
        return JSONResponse(content={"success": False, "message": f"La carpeta de students en la escuela '{escuela}' no existe."}, status_code=400)
    # Leer imagen enviada
    contents = await file.read()
    img_query = Image.open(BytesIO(contents))
    img_query_np = np.array(img_query)
    # Buscar todas las imágenes en la carpeta usando DeepFace.find
    if not os.listdir(ruta_students):
        return JSONResponse(content={"success": False, "message": "No hay imágenes en la carpeta students."}, status_code=404)
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
        return JSONResponse(content={"success": False, "message": "No se encontró ninguna coincidencia válida."}, status_code=404)
    mejor = df.iloc[0]
    porcentaje = round((1 - mejor['distance']) * 100, 2)
    if porcentaje <= 60:
        return JSONResponse(content={"success": False, "message": "No se encontró ninguna coincidencia con al menos 60% de similitud."}, status_code=404)
    resultado = {
        "success": True,
        "message": "Coincidencia encontrada en STUDENTS",
        "data": {
            "archivo": os.path.basename(mejor['identity']),
            "porcentaje_similitud": porcentaje,
            "tipo": "STUDENT"
        }
    }
    # Enviar datos a Laravel
    try:
        resp = requests.post(
            "http://127.0.0.1:8002/api1/entrada/create",
            json={
                "archivo": resultado["data"]["archivo"],
                "tipo": "STUDENT"
            },
            timeout=3
        )
        print("Laravel status:", resp.status_code)
        print("Laravel response:", resp.text)
    except Exception as e:
        print("Error al enviar a Laravel:", e)
    return resultado

#endregion

#region Escuelas
@app.post("/api2/crear/escuela")
async def crear_escuela(school_id: int = Form(...)):
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    if not os.path.isdir(ruta_escuela):
        try:
            os.makedirs(ruta_escuela, exist_ok=True)
            # Crear subcarpetas para cada tipo permitido
            for tipo in TIPOS_PERMITIDOS:
                os.makedirs(os.path.join(ruta_escuela, tipo), exist_ok=True)
            return JSONResponse(content={"success": True, "message": f"Escuela '{school_id}' creada en {ruta_escuela}"}, status_code=201)
        except Exception as e:
            return JSONResponse(content={"success": False, "message": f"Error al crear la escuela: {e}"}, status_code=500)
    return JSONResponse(content={"success": True, "message": f"Escuela '{school_id}' ya existe en {ruta_escuela}."}, status_code=200)

    #endregion

#region PRUEBA

@app.get("/api2/imagendurisima")
def image_maquiavelicamenteLetal():
    url = "https://static.wikia.nocookie.net/heroe/images/c/c4/AiAi_SMBBR.png/revision/latest?cb=20240710005028&path-prefix=es"
    response = requests.get(url)
    if response.status_code == 200:
        return StreamingResponse(BytesIO(response.content), media_type="image/png")
    return JSONResponse(content={"error": "No se pudo obtener la imagen"}, status_code=404)

# Nuevo endpoint para detectar si hay una persona en la imagen
@app.post("/api2/detecta")
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