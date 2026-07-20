
import requests, re, math, os, json
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import load_model


RUTA_HISTORICO = "historico_itinerarios.json"
PALABRAS_PROHIBIDAS = ["violencia", "drogas", "armas", "discriminación", "odio", "terrorismo"]

MAX_DOCS = 7
MAX_LEN = 200  # aumenta para cubrir itinerarios largos
VOCAB_SIZE = 10000
EMBED_DIM = 256  # más capacidad para vocabulario amplio
MIN_SCORE = 0.4
UMBRAL = 5
TOPK = 6 # número de mejores itinerarios a usar como ejemplos en el prompt

api_key = os.environ.get("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")
itinerarios = []
contador = 0


tokenizer = Tokenizer()


# Modelo policy simple (proxy)
model = tf.keras.Sequential([])

policy_optimizer = optimizers.Adam(learning_rate=0.001)


def cargar_itinerarios() -> list[dict]: # traer desde el archivo los itinerarios guardados (Formato JSON )
    if not os.path.exists(RUTA_HISTORICO):
        # Crear archivo vacío válido
        with open(RUTA_HISTORICO, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

    try: 
        with open(RUTA_HISTORICO, "r", encoding="utf-8") as f: # leer el archivo especificado en RUTA_HISTORICO
            contenido = f.read().strip()
            if not contenido:
                return [] # archivo vacío
            return json.loads(contenido) # devolver lista de diccionarios con texto, score, fecha
    except Exception as e: # manejar errores de lectura/parsing se reinicia el archivo por seguridad
        print(f"⚠️ Error al leer el archivo JSON: {e}. Se reinicia el histórico.")
        with open(RUTA_HISTORICO, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []


def guardar_itinerarios(itinerarios: list) -> None: # Guardar toda la lista de itinerarios en el archivo JSON desde memoria
    """Guarda toda la lista de itinerarios en el archivo JSON."""
    with open(RUTA_HISTORICO, "w", encoding="utf-8") as f:
        json.dump(itinerarios, f, ensure_ascii=False, indent=2)


def guardar_itinerario(texto: str, score: float) -> None: # guardar un nuevo itinerario en memoria
    global itinerarios
    """Añade un nuevo itinerario con su puntuación."""
    if score < MIN_SCORE: # descartar itinerarios con baja puntuación 
        feedback = input("¿Quieres dar una puntuación a este itinerario? (0-1, Enter para omitir): ")
        try:
            user_score = float(feedback)
            if 0.0 <= user_score <= 1.0:
                score = user_score
            else:
                print("⚠️ Puntuación inválida, se omite el itinerario.")
                return
        except ValueError:
            print("⚠️ Itinerario descartado por baja puntuación.")
            return
    
    # Evitar duplicados exactos
    if any(item["texto"] == texto for item in itinerarios):
        print("⚠️ Este itinerario ya existe en el histórico.")
        return
    
    #Nuevo itinerario con sus datos
    nuevo = {
        "texto": texto,
        "score": float(round(score, 4)),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    itinerarios.append(nuevo)

    print(f"✅ Itinerario añadido ({len(itinerarios)} total) | Puntuación: {score:.3f}")
    tokenizer.fit_on_texts([item["texto"] for item in itinerarios])  # actualizar el tokenizador con el nuevo texto

def cargar_modelo_policy() -> None: # cargar modelo preentrenado si existe
    global model, tokenizer, itinerarios

    if os.path.exists("modelo_policy.h5"):
        try:
            model = load_model("policy_model.keras")
            with open("tokenizer.json", "r", encoding="utf-8") as f:
                tokenizer = tokenizer_from_json(f.read())
                tokenizer.fit_on_texts([item["texto"] for item in itinerarios])
        except Exception as e:
            print(f"⚠️ Error al cargar el modelo de política: {e}")
    else:
        model = models.Sequential([
            layers.Input(shape=(MAX_LEN,)),      # secuencia tokenizada
            layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64),
            layers.LSTM(64),
            layers.Dense(1, activation='sigmoid')  # probabilidad de mantener/aceptar itinerario
        ])

        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>") 
        tokenizer.fit_on_texts([item["texto"] for item in itinerarios])

def save_model_policy() -> None: # guardar modelo y tokenizador
    global model, tokenizer

    try:
        model.save("policy_model.keras")
        with open("tokenizer.json", "w", encoding="utf-8") as f:
            f.write(tokenizer.to_json())
    except Exception as e:
        print(f"⚠️ Error al guardar el modelo de política: {e}")


def actualizar_politica(itinerarios: list[str]) -> None: # reentrenar el modelo con los históricos
    for texto in itinerarios:
        # tokenizar
        seq = tokenizer.texts_to_sequences([texto])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        pad = np.array(pad)
        
        # calcular recompensa
        r = evaluar_itinerario(texto)
        
        with tf.GradientTape() as tape:
            prob = model(pad, training=True)
            loss = -tf.math.log(prob + 1e-8) * r   # Policy gradient: gradiente ascendente hacia mayor recompensa
        
        grads = tape.gradient(loss, model.trainable_variables)
        policy_optimizer.apply_gradients(zip(grads, model.trainable_variables))

def evaluar_itinerario(itinerario: str) -> float:

    def check_palabras_prohibidas(itinerario: str) -> bool:
        texto_lower = itinerario.lower()
        for palabra in PALABRAS_PROHIBIDAS:
            if palabra in texto_lower:
                return False
        return True

    def check_longitud(itinerario: str, min_palabras=100, min_dias=1) -> bool:
        # Contar palabras
        num_palabras = len(itinerario.split())
        if num_palabras < min_palabras:
            return False
        
        # Contar días aproximados usando la palabra "Día"
        num_dias = itinerario.lower().count("día")
        if num_dias < min_dias:
            return False
        
        return True

    # Convertir texto a secuencia
    seq = tokenizer.texts_to_sequences([itinerario])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    # Predecir score con el modelo
    score = model.predict(pad, verbose=0)[0][0]

    # penalizar itinerarios demasiado cortos o con palabras prohibidas
    if not check_longitud(itinerario):
        score *= 0.5  # penalización fuerte
    if not check_palabras_prohibidas(itinerario):
        score *= 0.0  # descarte total

    return score


def descargar_guias(destino: str, max_docs: int = 5) -> list[dict]: # Descargar guías de Wikivoyage en inglés
    session = requests.Session() # sesión HTTP para reutilizar conexiones
    session.headers.update({"User-Agent": "viajes-simple/1.0"})
    resultados = []

    def _limpiar(txt: str) -> str: # limpiar el texto extraído
        txt = re.sub(r"\[\d+\]", "", txt)
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        return txt

    def _fetch_extract(title: str, lang: str): # obtener extracto de Wikivoyage
        base = f"https://{lang}.wikivoyage.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "wiki",
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
        r = session.get(base, params=params, timeout=15) # solicitud HTTP con timeout para no quedar esperando
        r.raise_for_status() # lanzar error si falla la solicitud
        pages = r.json().get("query", {}).get("pages", {}) # obtener páginas
        page = next(iter(pages.values()), {})  # tomar la primera página
        extract = page.get("extract", "") or "" # extraer texto
        return _limpiar(extract)

    search_titles = [
        destino,
        f"{destino} travel",
        f"{destino} guide",
        f"{destino} tourism"
    ]

    for title in search_titles:
        if len(resultados) >= max_docs: 
            break
        if len(texto) < 500:
            continue
        if any(texto == r["texto"] for r in resultados):
            continue
    
        try:
            texto, _  = _fetch_extract(title, lang="en")
            if len(texto) > 500:
                resultados.append({"url": f"https://en.wikivoyage.org/wiki/{title.replace(' ', '_')}", "texto": texto})
        except requests.RequestException:
            continue
    return resultados


def construir_contexto(guias: list[dict], max_chars: int = 8000) -> str: # Construir contexto a partir de guías descargadas

    if not guias:
        return "" # no hay guías
    per = max(1200, math.floor(max_chars / len(guias))) # distribuir caracteres entre guías
    partes = [] # construir partes del contexto
    for g in guias: # cada guía
        texto = (g.get("texto") or "")[:per] # limitar por caracteres
        partes.append(f"\n{texto}") # añadir al contexto
    contexto = "\n".join(partes) # unir todas las partes
    return contexto[:max_chars]# limitar contexto total


def abrir_conexion(contexto: str, destino: str, personas: str, dias: int, motivo: str, dinero: int): #### PSEUDO GENERADOR YA QUE LA API NO PERMITE ENTRENAMIENTO DIRECTO
    top_itinerarios = sorted(itinerarios, key=lambda x: x["score"], reverse=True)[:TOPK] # seleccionar los mejores itinerarios
    ejemplos = "\n\n".join([i["texto"] for i in top_itinerarios])

    try:         
        API_URL = "https://api.groq.com/openai/v1/chat/completions"
        headers = { 
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        } 
        system_msg = (
        "Eres un planificador de viajes experto. Respondes con itinerarios claros, "
        "agrupados por días, incluyendo comida típica y consejos locales. Si el contexto "
        "es limitado, explícitalo y evita inventar datos."
        )

        user_msg = f"""Usa este contexto para crear el itinerario.

        [CONTEXTO]
        {contexto}
        [MEJORES EJEMPLOS]
        {ejemplos}
        [TAREA]
        Genera un itinerario completo para "{destino}" de {personas} personas para {dias} dias {motivo} con {dinero} total de presupuesto, con actividades por días, comida típica y consejos locales.
        Incluye una sección final de “Consejos prácticos”.
        """
        payload = {
            "model": "llama-3.1-8b-instant",   # también: "llama-3.1-8b-instant" (más barato/rápido)
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2000,          # ajusta según lo que necesites
        }

        
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if not resp.ok:
            print(f"[HTTP {resp.status_code}] {resp.text[:400]}")
            return
        data = resp.json()
        texto = data["choices"][0]["message"]["content"]
        return texto

    except Exception as e:
        print("Ha ocurrido un error: " + str(e))


def recuperacion_informacion(destino: str) -> str:
    try:
        guias = descargar_guias(destino=destino, max_docs=MAX_DOCS)
        if not guias:
            return "No se encontraron guías."
        contexto = construir_contexto(guias, max_chars=8000)
        return contexto
    except Exception:
        return "No se encontró contexto relevante o hubo un error en la búsqueda."


def procesar(datos: list) -> int:
    global contador
    contexto = recuperacion_informacion(datos[0])
    itinerario = abrir_conexion(contexto, destino=datos[0], personas=datos[1],  dias=datos[2],  dinero=datos[3],  motivo=datos[4],  detalle=datos[5])
    if itinerario is not None: #Si no se genero el itinerario por cualquier cosa no seguimos el flujo
        contador += 1
        score = evaluar_itinerario(itinerario)
        print(itinerario)
        guardar_itinerario(itinerario, score)

        if contador % UMBRAL == 0:
           actualizar_politica([item["texto"] for item in itinerarios])  # reentrenar el modelo cada UMBRAL itinerarios


def main():
    global itinerarios
    loop = True
    first_time = True
    itinerarios = cargar_itinerarios()
    cargar_modelo_policy()  # cargar modelo preentrenado si existe

    while loop:
        if first_time:
            print("Buenos dias, vamos a ayudarte a planificar tu viaje.")
            first_time = False
        else:
            print("Quieres planificar otro viaje? (si/no/salir)")
            respuesta = input("> ")
            if respuesta.lower() == "no" or respuesta.lower() == "salir":
                print("Hasta luego!")
                break
        
        print("Introduce el destino del viaje:")
        destino = input("> ")
        print("¿Para cuántas personas es el viaje?")
        personas = input("> ")
        print("¿Cuántos días durará el viaje?")
        dias = int(input("> "))
        print("¿Cuál es el motivo del viaje? (ocio, negocios, aventura, etc.)")
        motivo = input("> ")
        print("¿Cuál es el presupuesto aproximado para el viaje?")
        dinero = int(input("> "))
        print("Quieres aportar algún detalle extra?")
        detalle = input("> ")
        if detalle.strip() or detalle != "no":
            detalle += f", {detalle.strip()}"

        try:
            personas = int(personas)
            dias = int(dias)
            dinero = int(dinero)
        except ValueError:
            print("⚠️ Por favor, ingresa un número válido para los días.")
            continue

        datos = [destino, personas, dias, dinero, motivo, detalle]
        procesar(datos)

    guardar_itinerarios(itinerarios) # Guardar todos los itinerarios al finalizar el programa
    save_model_policy()

if __name__ == "__main__":
    main()