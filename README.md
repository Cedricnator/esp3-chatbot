# esp3-chatbot
Desarrollar un asistente conversacional en Python que responda consultas de estudiantes y personal sobre normativa y reglamentos universitarios de la UFRO.

## Requisitos

- Sistema operativo: Linux / macOS / Windows (se recomienda Linux para despliegues).
- Python 3.10+ (se ha probado con 3.10 y 3.11).
- Dependencias listadas en `requirements.txt`.
- Espacio en disco para índices y checkpoints (varios MBs a GB según la colección).

## Instalación

1. Clonar el repositorio:

```
git clone https://github.com/Cedricnator/esp3-chatbot.git
```

2. Crear y activar un entorno virtual:

LINUX
```
python3 -m venv .venv
source .venv/bin/activate
```

WINDOWS
```bash
python3 -m venv .venv
.\.venv\Scripts\activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```


## Variables de entorno (.env)

El proyecto usa variables de entorno para proveedores LLM y configuraciones. Cree un archivo `.env` en la raíz con las claves necesarias. Este repositorio incluye `.env.example` con las claves mínimas:

- OPENAI_API_KEY — Clave para OpenAI (usada por `provider/chat_gpt.py`).
- DEEPSEEK_API_KEY — Clave para DeepSeek (usada por `provider/deepseek.py`).


Ejemplo mínimo (`.env.example`):

```
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
```

Nota de seguridad: No incluya claves en repositorios públicos. Use mecanismos seguros en producción.

## Uso de CLI y batch

El entrypoint principal es `src/main.py`. El parser de argumentos crea las opciones usadas en `Main.run()`; las opciones relevantes detectadas son:

- `-m | --message` — Texto de la consulta para ejecutar (string).
- `-r | --rag` — Habilita la ejecución RAG cuando no es `False` (por defecto puede ser True).
- `-e | --evaluation` — Si no es `False`, lanza el modo de evaluación por lotes usando `gold_set.json`.
- `-p | --provider` — Selecciona proveedor para chat: `chatgpt` o `deepseek`.

Ejemplos de uso:

1) Ejecutar RAG para una pregunta corta (usa `--message` y deja `--rag` habilitado):

```bash
python3 ./src/main.py --m "¿Cuál es el plazo de matrícula?" --rag True
```

2) Ejecutar evaluación por lotes (usa `--evaluation`):

```bash
python3 ./src/main.py --evaluation True
```

3) Enviar la misma consulta directamente a un proveedor (sin RAG) indicando `--provider`:

```bash
python -m src.main --message "Resumen del reglamento" --provider chatgpt
```

4) Usar DeepSeek provider:

```bash
python -m src.main --message "consulta" --provider deepseek
```

Para ver todas las opciones disponibles, abra `src/adapters/arg_parser.py`.

## Diagrama breve del pipeline RAG

1) Ingesta: extraer texto de PDFs → limpieza y chunking → almacenar chunks (parquet) y mapping.
2) Embeddings: calcular embeddings por chunk → construir índice (FAISS o similar).
3) Recuperación: para una consulta, recuperar top-K chunks relevantes.
4) Re-ranking (opcional): re-ranker sobre candidatos para ordenar mejor.
5) Generación: pasar contexto recuperado y prompt al LLM para generar respuesta.
6) Post-procesamiento: aplicar filtros de confidencialidad y políticas de abstención antes de devolver respuesta.


## Política de abstención y consideraciones (vigencia normativa y privacidad)

- Abstención: El asistente debe abstenerse de responder cuando:
  - No encuentra evidencia suficiente en los documentos indexados para sustentar una respuesta.
  - La consulta solicita asesoramiento legal, médico, financiero o decisiones con impacto legal que requieren juicio humano.
  - La pregunta pide datos personales sensibles o identificar a personas concretas.

- Mensaje de abstención sugerido:

  "No puedo responder con seguridad a esa consulta con la información disponible. Le recomiendo consultar el documento oficial o contactar a la unidad responsable."

- Vigencia normativa: Documente la fecha de vigencia encontrada en la fuente y preséntela junto a respuestas normativas. Si la vigencia es posterior o anterior a la consulta, advierta al usuario.

  Ejemplo de inclusión en respuestas:

  "Referencia: Reglamento de Admisión — vigencia 27-09-2025. Ver fuente para detalles y última versión." 

- Privacidad: Evite exponer información personal o sensible almacenada en las fuentes. Si las fuentes contienen datos personales, aplique redacción/anonimización y limite el contexto que se inserta en prompts.

## Tabla de trazabilidad (doc_id → URL / página / vigencia)

La siguiente tabla mapea los documentos usados en el índice a su ruta local, URL pública y fecha de vigencia (extraída de `data/sources.csv`).

| doc_id | nombre | ruta local | URL pública | fecha / vigencia |
|---|---|---|---|---|
| 01 | Reglamento Régimen de Estudios de Pregrado | `data/raw/01-Reglamento-de-Regimen-de-Estudios-2023.pdf` | https://www.ufro.cl/wp-content/uploads/2025/04/01-Reglamento-de-Regimen-de-Estudios-2023.pdf | 27-09-2025 |
| 02 | Reglamento de Admisión | `data/raw/02-Res-Ex-3542-2022-Reglamento-de-Admision-para-carreras-de-Pregrado.pdf` | https://www.ufro.cl/wp-content/uploads/2025/04/02-Res-Ex-3542-2022-Reglamento-de-Admision-para-carreras-de-Pregrado.pdf | 27-09-2025 |
| 03 | Reglamento de Obligaciones Financieras | `data/raw/03-resex-2022326308-obligaciones-financieras.pdf` | https://www.ufro.cl/wp-content/uploads/2025/04/03-resex-2022326308-obligaciones-financieras.pdf | 27-09-2025 |
| 04 | Reglamento de Convivencia Universitaria Estudiantil | `data/raw/04-Reglamento-Convivencia-rex.pdf` | https://www.ufro.cl/wp-content/uploads/2025/04/04-Reglamento-Convivencia-rex.pdf | 27-09-2025 |

Si necesita una exportación machine-readable (CSV/JSON), consulte `data/sources.csv` o solicite que añada `data/traceability.csv` con columnas `doc_id,nombre,path,url,fecha,vigencia`.

---

## Notas finales

1. Para pruebas rápidas, revise `lab/` (notebooks) que contienen ingest y pruebas con LLM.
2. Para cambiar proveedores LLM, revise `src/provider/` y actualice `.env` según corresponda.
