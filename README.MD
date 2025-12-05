# Analizador de Complejidad 🔍

Pequeña herramienta que analiza la complejidad computacional de algoritmos escritos en pseudocódigo. Incluye
análisis iterativo y recursivo, generación de sumatorias y visualizaciones (gráficas y árboles de recursión).

Características principales:

- Análisis de complejidad (O / Ω / Θ) para procedimientos.
- Detección de recursión y generación de ecuaciones de recurrencia.
- Conteo de operaciones básicas (OPD) y desglose por tipo.
- Visualizaciones interactivas con Streamlit.

## Requisitos

- Python 3.8+
- Recomendado crear un entorno virtual (venv / conda).

## Instalación

Desde la raíz del proyecto:

```bash
python -m venv venv
source venv/bin/activate    # o: venv\Scripts\activate en Windows (PowerShell: .\venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

Si quieres usar la validación con Groq (LLM), define tu API key como variable de entorno:

Opción rápida (archivo `.env` en la raíz):

```
GROQ_API_KEY=tu_api_key_aqui
```

Opción persistente en Windows (PowerShell):

```powershell
setx GROQ_API_KEY "tu_api_key_aqui"
```

## Ejecutar la aplicación

```bash
streamlit run ./app_complete.py
```

La interfaz carga ejemplos desde la carpeta `examples/` y permite analizar o inspeccionar operaciones básicas (OPD) sin ejecutar el análisis completo.

## Uso rápido

- Pega o escribe tu pseudocódigo en el textarea.
- Usa "Analizar" para ejecutar el análisis completo.
- Usa "Operaciones" para ver un conteo rápido de operaciones básicas.

## Estructura de carpetas

- `src/` : código fuente (parser, analizador, visualización, LLM)
- `examples/` : ejemplos de algoritmos (iterativos y recursivos)
- `app_complete.py` : interfaz principal con Streamlit

---

Proyecto educativo para análisis y visualización de algoritmos — diseñado para uso académico y pruebas locales.
