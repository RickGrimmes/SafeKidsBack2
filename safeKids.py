import asyncio
from typing import Dict, Set
from fastapi import FastAPI, File, UploadFile, Form, Request, Body, Query, WebSocket
import glob
import shutil
from fastapi.responses import JSONResponse, StreamingResponse
import queue
import threading
import requests
from io import BytesIO
from deepface import DeepFace
import os
from datetime import datetime
import json
os.environ["DEEPPFACE_HOME"] = "C:/Users/babaj/.deepface"

# Variable global para la ruta de imágenes
IMG_ROUTE = "C:/Users/babaj/Documents/9C/IMAGES"
# IMG_ROUTE = "/home/carlos/img"

# --- Variables globales para SSE ---
active_sse_connections: Dict[int, Set] = {}  # {school_id: {connection1, connection2}}
event_queues: Dict[int, queue.Queue] = {}  # {school_id: queue_of_events}

app = FastAPI()

# --- Constantes de tipos permitidos ---
TIPOS_PERMITIDOS = {"AUTHORIZEDS", "DIRECTOR", "GUARDIANS", "SECRETARY", "STUDENTS"}

# --- Variable global para student_mode con persistencia POR ESCUELA ---
STUDENT_MODES_FILE = "student_modes.json"

def load_student_modes():
    """Cargar student_modes desde archivo - diccionario por escuela"""
    try:
        if os.path.exists(STUDENT_MODES_FILE):
            with open(STUDENT_MODES_FILE, 'r') as f:
                data = json.load(f)
                return data
    except:
        pass
    return {}

def save_student_modes(modes_dict: dict):
    """Guardar student_modes en archivo"""
    try:
        with open(STUDENT_MODES_FILE, 'w') as f:
            json.dump(modes_dict, f, indent=2)
        return True
    except:
        return False

def get_school_student_mode(school_id: int):
    """Obtener student_mode de una escuela específica"""
    return student_modes.get(str(school_id), False)

def set_school_student_mode(school_id: int, mode: bool):
    """Establecer student_mode para una escuela específica"""
    global student_modes
    student_modes[str(school_id)] = mode
    return save_student_modes(student_modes)

# Cargar el estado inicial - ahora es un diccionario por escuela
student_modes = load_student_modes()

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
            deteccion = DeepFace.extract_faces(img_np, enforce_detection=False)  # MÁS PERMISIVO
        except Exception as e:
            return False, "No se detectó ningún rostro en la imagen. Por favor, sube una foto donde se vea claramente tu cara."
        if not isinstance(deteccion, list) or len(deteccion) < 1:  
            return False, "La imagen debe contener al menos un rostro visible."
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
        if w < min_side * 0.1 or h < min_side * 0.1: 
            return False, "El rostro es demasiado pequeño en la imagen. Acércate más a la cámara."
        # Validación de nitidez (enfoque)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        if laplacian_var < 15: 
            return False, "La imagen está borrosa. Por favor, asegúrate de que la foto esté bien enfocada."
        # Validación de brillo
        brightness = np.mean(img_gray)
        if brightness < 30:  
            return False, "La imagen está demasiado oscura. Busca mejor iluminación."
        if brightness > 240: 
            return False, "La imagen está demasiado clara o sobreexpuesta."
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

# --- Endpoint para subir imagen de director ---
@app.post("/api2/upload/director")
async def upload_director(school_id: int = File(...), id: str = File(...), firstName: str = File(...), lastName: str = File(...), file: UploadFile = File(...)):
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"success": False, "message": f"La escuela '{school_id}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(nombre_carpeta, "DIRECTOR", file, id, firstName, lastName)
    if ok:
        return {"success": True, "message": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"success": False, "message": msg}, status_code=400)

# --- Endpoint para subir imagen de secretary ---
@app.post("/api2/upload/secretary")
async def upload_secretary(school_id: int = File(...), id: str = File(...), firstName: str = File(...), lastName: str = File(...), file: UploadFile = File(...)):
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"success": False, "message": f"La escuela '{school_id}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(nombre_carpeta, "SECRETARY", file, id, firstName, lastName)
    if ok:
        return {"success": True, "message": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"success": False, "message": msg}, status_code=400)

# --- Endpoint para subir imagen de student ---
@app.post("/api2/upload/students")
async def upload_students(school_id: int = File(...), id: str = File(...), firstName: str = File(...), lastName: str = File(...), file: UploadFile = File(...)):
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"success": False, "message": f"La escuela '{school_id}' no existe. Primero debe crear la carpeta de la escuela."}, status_code=400)
    ok, msg = guardar_imagen(nombre_carpeta, "STUDENTS", file, id, firstName, lastName)
    if ok:
        return {"success": True, "message": f"Imagen guardada en {msg}"}
    return JSONResponse(content={"success": False, "message": msg}, status_code=400)

#endregion  

#region UpdateImages

# --- Endpoint para actualizar/reemplazar foto de estudiante ---
@app.post("/api2/update/student/photo")
async def update_student_photo(school_id: int = File(...), student_photo: str = File(...), file: UploadFile = File(...)):
    from PIL import Image
    import numpy as np
    import cv2
    
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    
    # Verificar que la escuela existe
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"success": False, "message": f"La escuela '{school_id}' no existe."}, status_code=400)
    
    # Ruta de la carpeta STUDENTS
    ruta_students = os.path.join(ruta_escuela, "STUDENTS")
    if not os.path.isdir(ruta_students):
        return JSONResponse(content={"success": False, "message": f"La carpeta STUDENTS de la escuela '{school_id}' no existe."}, status_code=400)
    
    # Ruta completa del archivo a actualizar
    ruta_archivo_actual = os.path.join(ruta_students, student_photo)
    
    # Verificar que el archivo existe
    if not os.path.exists(ruta_archivo_actual):
        return JSONResponse(content={"success": False, "message": f"El archivo '{student_photo}' no existe en la carpeta STUDENTS."}, status_code=404)
    
    try:
        # Leer y validar la nueva imagen
        contents = file.file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_np = np.array(img)
        
        # Validar que haya exactamente un rostro
        try:
            deteccion = DeepFace.extract_faces(img_np, enforce_detection=False)
        except Exception as e:
            return JSONResponse(content={"success": False, "message": "No se detectó ningún rostro en la imagen. Por favor, sube una foto donde se vea claramente la cara."}, status_code=400)
        
        if not isinstance(deteccion, list) or len(deteccion) < 1:  
            return JSONResponse(content={"success": False, "message": "La imagen debe contener al menos un rostro visible."}, status_code=400)
        
        # Validación de tamaño mínimo del rostro
        face = deteccion[0]
        region = face.get('facial_area') or face.get('region')
        if not region:
            return JSONResponse(content={"success": False, "message": "No se pudo determinar el área del rostro."}, status_code=400)
        
        w = region.get('w') or region.get('width')
        h = region.get('h') or region.get('height')
        if not w or not h:
            return JSONResponse(content={"success": False, "message": "No se pudo determinar el tamaño del rostro."}, status_code=400)
        
        min_side = min(img_np.shape[0], img_np.shape[1])
        if w < min_side * 0.1 or h < min_side * 0.1: 
            return JSONResponse(content={"success": False, "message": "El rostro es demasiado pequeño en la imagen. Acércate más a la cámara."}, status_code=400)
        
        # Validación de nitidez (enfoque)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        if laplacian_var < 15: 
            return JSONResponse(content={"success": False, "message": "La imagen está borrosa. Por favor, asegúrate de que la foto esté bien enfocada."}, status_code=400)
        
        # Validación de brillo
        brightness = np.mean(img_gray)
        if brightness < 30:  
            return JSONResponse(content={"success": False, "message": "La imagen está demasiado oscura. Busca mejor iluminación."}, status_code=400)
        if brightness > 240: 
            return JSONResponse(content={"success": False, "message": "La imagen está demasiado clara o sobreexpuesta."}, status_code=400)
        
        # ELIMINAR archivo anterior
        try:
            os.remove(ruta_archivo_actual)
            print(f"[UPDATE] Archivo anterior eliminado: {ruta_archivo_actual}")
        except Exception as e:
            return JSONResponse(content={"success": False, "message": f"Error al eliminar archivo anterior: {e}"}, status_code=500)
        
        # GUARDAR nueva imagen con el mismo nombre
        try:
            img.save(ruta_archivo_actual, format="JPEG", quality=80, optimize=True)
            print(f"[UPDATE] Nueva imagen guardada: {ruta_archivo_actual}")
        except Exception as e:
            return JSONResponse(content={"success": False, "message": f"Error al guardar nueva imagen: {e}"}, status_code=500)
        
        return {
            "success": True, 
            "message": f"Foto del estudiante actualizada exitosamente: {student_photo}",
            "data": {
                "school_id": school_id,
                "archivo_actualizado": student_photo,
                "ruta_completa": ruta_archivo_actual,
                "updated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Error al procesar la imagen: {e}"}, status_code=500)

#endregion

#region BuscaPersonas

# --- Endpoint para buscar el guardian más parecido en una escuela, si no, entonces busca al authorized ---
@app.post("/api2/busca/guardianAuthPeople")
async def busca_guardian(school_id: int = File(...), file: UploadFile = File(...)):
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
    nombre_carpeta = str(school_id)
    ruta_guardians = os.path.join(IMG_ROUTE, nombre_carpeta, "GUARDIANS")
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
                if porcentaje >= 70:
                    resultado = {"success": True, "message": "Coincidencia encontrada en GUARDIANS", "data": {"archivo": os.path.basename(row['identity']), "porcentaje_similitud": porcentaje, "tipo": "GUARDIAN"}}
    
                    # NUEVO: Enviar via SSE
                    await send_sse_event(school_id, {
                        "type": "recognition_result",
                        "event": "guardian_found", 
                        "data": resultado,
                        "timestamp": datetime.now().isoformat()
                    })
    
                    return resultado
                
    # --- Buscar en AUTHORIZEDS ---
    ruta_authorizeds = os.path.join(IMG_ROUTE, nombre_carpeta, "AUTHORIZEDS")
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
                if porcentaje >= 70:
                    resultado = {"success": True, "message": "Coincidencia encontrada en AUTHORIZEDS", "data": {"archivo": os.path.basename(row['identity']), "porcentaje_similitud": porcentaje, "tipo": "AUTHORIZED"}}
                    
                    # NUEVO: Enviar via SSE
                    await send_sse_event(school_id, {
                        "type": "recognition_result",
                        "event": "authorized_found",
                        "data": resultado,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    return resultado
    
    # Si no encontró nada en ninguno
    error_result = {"success": False, "message": "No se encontró ninguna coincidencia válida en guardians ni authorizeds con al menos 70% de similitud.", "school_id": school_id}
    
    # NUEVO: Enviar via SSE para error
    await send_sse_event(school_id, {
        "type": "recognition_result",
        "event": "guardian_not_found",
        "data": error_result,
        "timestamp": datetime.now().isoformat()
    })
    
    return JSONResponse(content=error_result, status_code=404)

# --- Endpoint para buscar el student más parecido en una escuela OSEA ENTRADA ---
@app.post("/api2/busca/student")
async def busca_student(school_id: int = File(...), file: UploadFile = File(...)):
    import numpy as np
    from PIL import Image
    from deepface import DeepFace
    import glob
    nombre_carpeta = str(school_id)
    ruta_students = os.path.join(IMG_ROUTE, nombre_carpeta, "STUDENTS")
    if not os.path.isdir(ruta_students):
        return JSONResponse(content={"success": False, "message": f"La carpeta de students en la escuela '{school_id}' no existe."}, status_code=400)
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
        error_result = {"success": False, "message": "No se encontró ninguna coincidencia válida.", "school_id": school_id}
        
        # NUEVO: Enviar via SSE para error
        await send_sse_event(school_id, {
            "type": "recognition_result",
            "event": "student_not_found",
            "data": error_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return JSONResponse(content=error_result, status_code=404)
    
    mejor = df.iloc[0]
    porcentaje = round((1 - mejor['distance']) * 100, 2)
    
    if porcentaje <= 60:
        error_result = {"success": False, "message": "No se encontró ninguna coincidencia con al menos 60% de similitud.", "school_id": school_id}
        
        # NUEVO: Enviar via SSE para error
        await send_sse_event(school_id, {
            "type": "recognition_result", 
            "event": "student_low_confidence",
            "data": error_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return JSONResponse(content=error_result, status_code=404)
    resultado = {
        "success": True,
        "message": "Coincidencia encontrada en STUDENTS",
        "data": {
            "archivo": os.path.basename(mejor['identity']),
            "porcentaje_similitud": porcentaje,
            "tipo": "STUDENT"
        }
    }

    # NUEVO: Enviar via SSE
    await send_sse_event(school_id, {
        "type": "recognition_result",
        "event": "student_found",
        "data": resultado,
        "timestamp": datetime.now().isoformat()
    })

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

#region Crear Escuelas
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

#region Eliminar 
@app.post("/api2/eliminar/foto")
async def eliminar_foto(
    school_id: int = Body(...),
    role_type: str = Body(...),
    profilePhoto: str = Body(...)
):
    tipo = role_type.upper()
    if tipo not in TIPOS_PERMITIDOS:
        return JSONResponse(content={"success": False, "message": f"El tipo '{role_type}' no es permitido."}, status_code=400)
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"success": False, "message": f"La escuela '{school_id}' no existe."}, status_code=404)
    ruta_foto = os.path.join(ruta_escuela, tipo, profilePhoto)
    if not os.path.isfile(ruta_foto):
        return JSONResponse(content={"success": False, "message": f"La foto '{profilePhoto}' no existe en la escuela '{school_id}' y tipo '{tipo}'."}, status_code=404)
    try:
        os.remove(ruta_foto)
        return {"success": True, "message": f"Foto '{profilePhoto}' eliminada exitosamente de la escuela '{school_id}' y tipo '{tipo}'."}
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Error al eliminar la foto: {e}"}, status_code=500)

@app.post("/api2/eliminar/escuela")
async def eliminar_escuela(school_id: int = Body(...)):
    nombre_carpeta = str(school_id)
    ruta_escuela = os.path.join(IMG_ROUTE, nombre_carpeta)
    if not os.path.isdir(ruta_escuela):
        return JSONResponse(content={"success": False, "message": f"La escuela '{school_id}' no existe."}, status_code=404)
    # Revisar cada subcarpeta
    for subcarpeta in os.listdir(ruta_escuela):
        ruta_sub = os.path.join(ruta_escuela, subcarpeta)
        if os.path.isdir(ruta_sub):
            archivos_jpg = glob.glob(os.path.join(ruta_sub, "*.jpg"))
            if archivos_jpg:
                return JSONResponse(content={"success": False, "message": f"No se puede eliminar la escuela '{school_id}' porque la carpeta '{subcarpeta}' tiene imágenes de usuarios."}, status_code=400)
    # Si no hay imágenes, eliminar toda la escuela
    try:
        shutil.rmtree(ruta_escuela)
        return {"success": True, "message": f"Escuela '{school_id}' eliminada exitosamente con todas sus carpetas."}
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Error al eliminar la escuela: {e}"}, status_code=500)

#endregion

#region STUDENT MODE MANAGEMENT

# --- Endpoint para toggle student_mode POR ESCUELA ---
@app.post("/api2/toggle-student-mode")
async def toggle_student_mode(school_id: int = Body(...)):
    try:
        # Obtener el modo actual de esa escuela específica
        current_mode = get_school_student_mode(school_id)
        
        # Cambiar al valor contrario
        new_mode = not current_mode
        
        # Guardar el nuevo estado para esa escuela
        if set_school_student_mode(school_id, new_mode):
            resultado = {
                "success": True,
                "message": f"Student mode {'activado' if new_mode else 'desactivado'} exitosamente para escuela {school_id}",
                "data": {
                    "school_id": school_id,
                    "student_mode": new_mode,
                    "previous_mode": current_mode,
                    "updated_at": datetime.now().isoformat()
                }
            }
            
            # NUEVO: Enviar via SSE
            await send_sse_event(school_id, {
                "type": "student_mode_changed",
                "event": "mode_toggle",
                "data": resultado,
                "timestamp": datetime.now().isoformat()
            })
            
            return resultado
        else:
            return JSONResponse(content={
                "success": False,
                "message": "Error al guardar el estado"
            }, status_code=500)
            
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "message": f"Error al cambiar student mode: {e}"
        }, status_code=500)

#endregion

#region SSE EVENTS

@app.get("/api2/events/{school_id}")
async def sse_events(school_id: int):
    async def event_stream():
        # Agregar esta conexión a las activas
        if school_id not in active_sse_connections:
            active_sse_connections[school_id] = set()
        
        # Crear cola de eventos para esta escuela si no existe
        if school_id not in event_queues:
            event_queues[school_id] = queue.Queue()
        
        # Identificador único para esta conexión
        connection_id = id(asyncio.current_task())
        active_sse_connections[school_id].add(connection_id)
        
        try:
            print(f"[SSE] Cliente conectado a escuela {school_id}")
            
            # Enviar evento de conexión exitosa
            yield f"data: {json.dumps({'type': 'connected', 'school_id': school_id, 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Loop principal para enviar eventos
            while True:
                try:
                    # Verificar si hay eventos en la cola (no bloqueante)
                    try:
                        event = event_queues[school_id].get_nowait()
                        yield f"data: {json.dumps(event)}\n\n"
                        print(f"[SSE] Evento enviado a escuela {school_id}: {event['type']}")
                    except queue.Empty:
                        # Si no hay eventos, enviar ping para mantener conexión viva
                        yield f"data: {json.dumps({'type': 'ping', 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                    # Esperar un poco antes de verificar nuevos eventos
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"[SSE] Error enviando evento a escuela {school_id}: {e}")
                    break
                
        except Exception as e:
            print(f"[SSE] Error en conexión de escuela {school_id}: {e}")
        finally:
            # Limpiar conexión al desconectarse
            if school_id in active_sse_connections:
                active_sse_connections[school_id].discard(connection_id)
                if not active_sse_connections[school_id]:
                    del active_sse_connections[school_id]
            print(f"[SSE] Cliente desconectado de escuela {school_id}")
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Función para enviar eventos a clientes conectados
async def send_sse_event(school_id: int, event_data: dict):
    """Enviar evento SSE a todos los clientes conectados de una escuela"""
    if school_id in active_sse_connections and active_sse_connections[school_id]:
        print(f"[SSE] Enviando evento a {len(active_sse_connections[school_id])} clientes de escuela {school_id}")
        
        # Crear cola de eventos para esta escuela si no existe
        if school_id not in event_queues:
            event_queues[school_id] = queue.Queue()
        
        # Agregar evento a la cola
        event_queues[school_id].put(event_data)
        print(f"[SSE] Evento '{event_data.get('type', 'unknown')}' agregado a cola de escuela {school_id}")
    else:
        print(f"[SSE] No hay clientes conectados para escuela {school_id} - evento descartado")

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
async def detecta_persona(school_id: int, file: UploadFile = File(...)):
    try:
        # Leer el contenido del archivo
        contents = await file.read()
        
        # Verificar que hay contenido
        if not contents:
            return {"persona": False, "mensaje": "No se recibió ninguna imagen", "school_id": school_id}
        
        from PIL import Image
        import numpy as np
        import tempfile
        import os
        
        # Crear archivo temporal para la imagen
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(contents)
            tmp.flush()  # Asegurar que se escriba al disco
            tmp_path = tmp.name
        
        try:
            # Abrir imagen desde archivo temporal
            img = Image.open(tmp_path).convert("RGB")
            img_np = np.array(img)

            # Aquí haces tu análisis con DeepFace
            DeepFace.analyze(img_np, actions=["emotion"], enforce_detection=False)

            return {"persona": True, "mensaje": "Sí hay una persona en la imagen", "school_id": school_id}
        finally:
            # Limpiar archivo temporal
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
    except Exception as e:
        return {"persona": False, "mensaje": f"No se detectó ningún rostro: {str(e)}", "school_id": school_id}
    
#endregion

#region PUENTE DEL ESP32CAM HACIA AQUÍ

# --- Endpoint UNIFICADO para routing por hora ---
@app.post("/api2/crear/entrada")
async def crear_entrada(request: Request, school_id: int, tipo_salida: str = None):
    from datetime import datetime
    from fastapi import UploadFile
    from io import BytesIO
    
    # Leer imagen enviada directamente del body
    contents = await request.body()
    
    # Verificar que hay contenido
    if not contents:
        return JSONResponse(content={
            "success": False, 
            "message": "No se recibió ninguna imagen", 
            "school_id": school_id
        }, status_code=400)
    
    # ROUTING POR HORA
    hora_actual = datetime.now().hour
    
    if 7 <= hora_actual <= 12:  # De 7 a 12 = ENTRADA (registrar_entrada)
        print(f"[ROUTING] Hora {hora_actual} - Llamando a registrar_entrada()")
        
        # Crear un nuevo Request object con el mismo body
        from fastapi import Request
        from starlette.requests import Request as StarletteRequest
        
        # Crear un request mock para pasar a registrar_entrada
        class MockRequest:
            async def body(self):
                return contents
        
        mock_request = MockRequest()
        
        resultado = await registrar_entrada(mock_request, school_id)
        
        # Agregar info de routing al resultado
        if isinstance(resultado, dict):
            resultado["routing_info"] = {
                "hora": hora_actual,
                "metodo_usado": "registrar_entrada",
                "motivo": "Horario de entrada (7-12)"
            }
        
        return resultado
        
    else:  # Cualquier otra hora = SALIDA (buscar guardian/authorized O buscar student)
        # Convertir body a UploadFile para ambos métodos
        fake_file = UploadFile(
            filename="image.jpg",
            file=BytesIO(contents),
            content_type="image/jpeg"
        )
        
        # DECIDIR QUÉ MÉTODO USAR EN SALIDA BASADO EN STUDENT_MODE DE LA ESCUELA
        school_student_mode = get_school_student_mode(school_id)
        
        # Si student_mode está activo para esta escuela, automáticamente usar "student"
        if school_student_mode:
            tipo_salida_efectivo = "student"
            print(f"[ROUTING] Hora {hora_actual} - STUDENT_MODE ACTIVO en escuela {school_id} - Forzando tipo_salida='student'")
        else:
            # Si no está activo, usar el parámetro original o default "guardian"
            tipo_salida_efectivo = tipo_salida or "guardian"
            print(f"[ROUTING] Hora {hora_actual} - STUDENT_MODE INACTIVO en escuela {school_id} - Usando tipo_salida='{tipo_salida_efectivo}'")
        
        # Ejecutar según el tipo efectivo
        if tipo_salida_efectivo == "student":
            print(f"[ROUTING] Llamando a busca_student() para SALIDA")
            resultado = await busca_student(school_id, fake_file)
            metodo_usado = "busca_student"
            
        else:  # Por defecto "guardian"
            print(f"[ROUTING] Llamando a busca_guardian() para SALIDA")
            resultado = await busca_guardian(school_id, fake_file)
            metodo_usado = "busca_guardian"
        
        # Agregar info de routing al resultado
        if isinstance(resultado, dict):
            resultado["routing_info"] = {
                "hora": hora_actual,
                "metodo_usado": metodo_usado,
                "motivo": "Horario de salida (fuera de 7-12)",
                "school_id": school_id,
                "student_mode_activo": school_student_mode,
                "tipo_salida_original": tipo_salida,
                "tipo_salida_efectivo": tipo_salida_efectivo
            }
        
        return resultado
    
    # Nunca debería llegar aquí
    return JSONResponse(content={"error": "Error de routing"}, status_code=500)

#endregion

#region ENTRADAS/SALIDAS

# --- Endpoint para registrar entrada ---
@app.post("/api2/registrar-entrada")
async def registrar_entrada(request: Request, school_id: int):
    import numpy as np
    from PIL import Image
    from deepface import DeepFace
    import glob
    import tempfile
    
    nombre_carpeta = str(school_id)
    ruta_students = os.path.join(IMG_ROUTE, nombre_carpeta, "STUDENTS")
    
    if not os.path.isdir(ruta_students):
        return JSONResponse(content={"success": False, "message": f"La carpeta de students en la escuela '{school_id}' no existe."}, status_code=400)
    
    # Leer imagen enviada directamente del body
    contents = await request.body()
    
    # Verificar que hay contenido
    if not contents:
        return JSONResponse(content={
            "success": False, 
            "message": "No se recibió ninguna imagen", 
            "school_id": school_id
        }, status_code=400)
    
    # Crear archivo temporal para la imagen (igual que detecta)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(contents)
        tmp.flush()  # Asegurar que se escriba al disco
        tmp_path = tmp.name
    
    try:
        # Abrir imagen desde archivo temporal
        img_query = Image.open(tmp_path).convert("RGB")
        img_query_np = np.array(img_query)
    except Exception as e:
        os.remove(tmp_path)  # Limpiar si hay error
        return JSONResponse(content={
            "success": False, 
            "message": f"Error al procesar la imagen: {str(e)}", 
            "school_id": school_id
        }, status_code=400)
    
    # Buscar todas las imágenes en la carpeta usando DeepFace.find
    if not os.listdir(ruta_students):
        os.remove(tmp_path)  # Limpiar archivo temporal
        return JSONResponse(content={"success": False, "message": "No hay imágenes de estudiantes registrados."}, status_code=404)
    
    # Ya tenemos el archivo temporal, usar directamente para DeepFace
    try:
        df = DeepFace.find(img_path=tmp_path, db_path=ruta_students, enforce_detection=False, model_name='Facenet')
    finally:
        os.remove(tmp_path)
    
    # DeepFace.find puede devolver un DataFrame o una lista de DataFrames
    if isinstance(df, list):
        df = df[0]
    
    if df.empty:
        return JSONResponse(content={
            "success": False, 
            "message": "Estudiante no reconocido - No se encontró coincidencia para ENTRADA.",
            "evento": "ENTRADA",
            "school_id": school_id
        }, status_code=404)
    
    mejor = df.iloc[0]
    porcentaje = round((1 - mejor['distance']) * 100, 2)
    
    if porcentaje <= 60:
        return JSONResponse(content={
            "success": False, 
            "message": f"Estudiante no reconocido - Similitud muy baja: {porcentaje}% para ENTRADA.",
            "evento": "ENTRADA",
            "school_id": school_id
        }, status_code=404)
    
    resultado = {
        "success": True,
        "message": f"¡ENTRADA REGISTRADA! Estudiante reconocido con {porcentaje}% de similitud",
        "data": {
            "archivo": os.path.basename(mejor['identity']),
            "porcentaje_similitud": porcentaje,
            "tipo": "STUDENT",
            "evento": "ENTRADA",
            "school_id": school_id
        }
    }
    
    # NUEVO: Enviar via SSE
    await send_sse_event(school_id, {
        "type": "recognition_result",
        "event": "student_entry",
        "data": resultado,
        "timestamp": datetime.now().isoformat()
    })
    
    # Enviar datos a Laravel - CREAR ENTRADA
    try:
        resp = requests.post(
            "http://127.0.0.1:8002/api1/entrada/check-in",
            json={
                "archivo": resultado["data"]["archivo"],
                "tipo": "STUDENT",
                "school_id": school_id,
            },
            timeout=3
        )
        print(f"Laravel CREAR ENTRADA - Status: {resp.status_code}")
        print(f"Laravel CREAR ENTRADA - Response: {resp.text}")
    except Exception as e:
        print(f"Error al crear ENTRADA en Laravel: {e}")
    
    return resultado

# --- Endpoint para registrar salida de estudiante ---
@app.post("/api2/registrar-salida")
async def registrar_salida(request: Request, school_id: int):
    import numpy as np
    from PIL import Image
    from deepface import DeepFace
    import glob
    import tempfile
    
    nombre_carpeta = str(school_id)
    ruta_students = os.path.join(IMG_ROUTE, nombre_carpeta, "STUDENTS")
    
    if not os.path.isdir(ruta_students):
        return JSONResponse(content={"success": False, "message": f"La carpeta de students en la escuela '{school_id}' no existe."}, status_code=400)
    
    # Leer imagen enviada directamente del body
    contents = await request.body()
    
    # Verificar que hay contenido
    if not contents:
        return JSONResponse(content={
            "success": False, 
            "message": "No se recibió ninguna imagen", 
            "school_id": school_id
        }, status_code=400)
    
    # Crear archivo temporal para la imagen (igual que detecta)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(contents)
        tmp.flush()  # Asegurar que se escriba al disco
        tmp_path = tmp.name
    
    try:
        # Abrir imagen desde archivo temporal
        img_query = Image.open(tmp_path).convert("RGB")
        img_query_np = np.array(img_query)
    except Exception as e:
        os.remove(tmp_path)  # Limpiar si hay error
        return JSONResponse(content={
            "success": False, 
            "message": f"Error al procesar la imagen: {str(e)}", 
            "school_id": school_id
        }, status_code=400)
    
    # Buscar todas las imágenes en la carpeta usando DeepFace.find
    if not os.listdir(ruta_students):
        os.remove(tmp_path)  # Limpiar archivo temporal
        return JSONResponse(content={"success": False, "message": "No hay imágenes de estudiantes registrados."}, status_code=404)
    
    # Ya tenemos el archivo temporal, usar directamente para DeepFace
    try:
        df = DeepFace.find(img_path=tmp_path, db_path=ruta_students, enforce_detection=False, model_name='Facenet')
    finally:
        os.remove(tmp_path)
    
    # DeepFace.find puede devolver un DataFrame o una lista de DataFrames
    if isinstance(df, list):
        df = df[0]
    
    if df.empty:
        return JSONResponse(content={
            "success": False, 
            "message": "Estudiante no reconocido - No se encontró coincidencia para SALIDA.",
            "evento": "SALIDA",
            "school_id": school_id
        }, status_code=404)
    
    mejor = df.iloc[0]
    porcentaje = round((1 - mejor['distance']) * 100, 2)
    
    if porcentaje <= 60:
        return JSONResponse(content={
            "success": False, 
            "message": f"Estudiante no reconocido - Similitud muy baja: {porcentaje}% para SALIDA.",
            "evento": "SALIDA",
            "school_id": school_id
        }, status_code=404)
    
    resultado = {
        "success": True,
        "message": f"¡SALIDA REGISTRADA! Estudiante reconocido con {porcentaje}% de similitud",
        "data": {
            "archivo": os.path.basename(mejor['identity']),
            "porcentaje_similitud": porcentaje,
            "tipo": "STUDENT",
            "evento": "SALIDA",
            "school_id": school_id
        }
    }
    
    # NUEVO: Enviar via SSE
    await send_sse_event(school_id, {
        "type": "recognition_result",
        "event": "student_exit",
        "data": resultado,
        "timestamp": datetime.now().isoformat()
    })
    
    # Enviar datos a Laravel - CREAR SALIDA
    try:
        resp = requests.post(
            "http://127.0.0.1:8002/api1/salida/create",  # Diferente endpoint para salidas
            json={
                "archivo": resultado["data"]["archivo"],
                "tipo": "STUDENT",
                "evento": "SALIDA",
                "school_id": school_id,
                "porcentaje": porcentaje
            },
            timeout=3
        )
        print(f"Laravel CREAR SALIDA - Status: {resp.status_code}")
        print(f"Laravel CREAR SALIDA - Response: {resp.text}")
    except Exception as e:
        print(f"Error al crear SALIDA en Laravel: {e}")
    
    return resultado

#endregion

#region ACTUALIZAR IMÁGENES


#endregion


# En la entrada, el niño solo se escanea, la cámara ya tiene el método que es y detecta la hora, si es para entrar entonces este cae en el registrar entrada y ya con eso se completa, hay que ver si sí se cumple ese ciclo pero listo hasta ahí
# En la salida, se usarán 2 métodos al menos, el cliente inicia la salida, debe enviar que sea salida, por lo que se puede tanto por la hora como tengo como porque manualmente ya le puso la secretaria salida, prioridad a la secretaria, entonces, primero ella debe marcar que es salida, ella comienza la comunicación

#PROCESO DE SALIDA DE FIN A INICIO
#CLIENTE GUARDA SALIDA Y LE ENVÍA A LARAVEL PARA CREAR REGISTRO
#SE OBTIENEN LOS REGISTROS DE CADA NIÑO PARA VER SI SÍ ES, SE ESCANEAN Y AQUÍ EN PYTHON SE DEVUELVE UN RESULTADO QUE VA A INTERPRETAR LARAVEL Y MOSTRAR AL CLIENTE
#NIÑO SE ESCANEA Y LA CÁMARA LE DICE A PYTHON QUE ES PARA ESTUDIANTES DE LEY
#SE OBTIENE EL REGISTRO DEL TUTOR O AUTORIZADO SI SÍ ES, Y PUES MUESTRA LOS CHIQUILLOS ASOCIADOS
#SE TOMA FOTO AL SEÑOR, YA SABE QUE ES TUTOR O AUTORIZADO POR SER SALIDA, COMO QUIERA PYTHON USA EL MISMO MÉTODO PARA AMBOS, POR LO QUE LO BUSCA ASÍ
#ENTORNO SE PREPARA PARA SALIDA, PORQUE EL CLIENTE DIJO QUE ES SALIDA
#el cliente debe mandar una señal para avisar que es salida
#CLIENTE PONE QUE YA TOCA SALIDA PARA EMPEZAR A ESCANEAR

#AL INICIAR SALIDA, EL CLIENTE SOLO LE PIDE A PYTHON EL BUSCAR GUARDIAN Y TOMA EL VALOR DEL REGISTRO OBTENIDO EN LA FOTO DE LA CAMARA, OSEA, TOMA FOTO A ESA HORA Y PYTHON RESPONDE PARA EL CLIENTE, ESA RESPUESTA DEBO MAPEARLA PARA QUE LE DÉ EL RESULTADO DE ESE FULANO, EMPIEZA AQUÍ CREO Y LUEGO VA HASTA LARAVEL PARA TENER EL TUTOR O FULANO COMPLETO Y SUS RELACIONES, YA ESO QUE LO GUARDE EN ALGÚN LADO PARA SEGUIR LA OPERACIÓN AHORA CON LOS STUDENTS, IGUAL Y UN BOTÓN QUE DIGA TIPO, CONFIRMAR QUE SÍ ES ESTE EL TUTOR? SI SÍ, QUE GUARDE EN ALGUN LADO QUE AHORA SIGUEN SUS STUDENTS Y SE LO MANDA A PYTHON SUPONGO PARA QUE ESTE SEPA OK DESDE AQUÍ YA TOCA PURO STUDENT PARA QUE LAS PROXIMAS FOTOS SEPA HACIA DÓNDE VA, LUEGO DE ESO QUE USE EL DE BUSCA STUDENT Y EL RESULTADO IGUAL QUE LO MUESTRE Y LO DEBO MAPEAR PARA VER EL RESULTADO, CREO QUE EMPIEZA AQUÍ Y SE VA A LARAVEL PARA RETORNAR ESE OBJETO, YA PARA IR LLENANDO EL ARREGLO DE REGISTRO DE SALIDA, Y CUANDO YA SE ACABA EL REGISTRO QUE LO ENVÍE A LARAVEL COMO FORMULARIO Y FIN, LE QUITA EL VALOR ESTE DE QUE DICE EL CLIENTE QUE YA TOCA PURO STUDENT, SE LO QUITAMOS PARA QUE VUELVA A SOLO TUTORES Y REPETIR CICLO, LA CÁMARA NO SABE NADA, SOLO TOMA FOTOS

#POR HORARIO, SABE QUE VA A BUSCAR AL GUARDIAN PARA LA SALIDA, LA COSA ES, UNA VEZ RETORNADO EL RESULTADO, PIENSO, DEBERÍA DE DECIR EL CLIENTE QUE AH SÍ ES, Y CON ESO DE ALGUNA MANERA ENVIARME A PYTHON COMO TAL UNA VARIABLE GLOBAL, STUDENT_MODE, SI EL TUTOR OBTENIDO ES Y ASÍ PUES ENVIA ESA SEÑAL PARA ACÁ Y SE GUARDA EN LA VARIABLE, PARA QUE DESDE AHÍ PYTHON SEPA QUE EN CREAR ENTRADA, SI ESE STUDENT MODE ESTÁ ACTIVO, ENTONCES NOMÁS USA EL MÉTODO DE BUSCAR STUDENTS, Y ASÍ SE VA, LUEGO CUANDO YA EL CLIENTE ENVIE EL REGISTRO COMPLETO A LARAVEL, QUE ESTE TAMBIÉN LE DIGA AL PYTHON DE QUE OYE CARNAL, DESACTIVA STUDENT_MODE Y ASÍ LISTO