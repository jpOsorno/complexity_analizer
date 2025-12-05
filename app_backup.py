"""
Analizador de Complejidad Computacional
=======================================

Interfaz web para analizar la complejidad de algoritmos en pseudocódigo.

Ejecutar: streamlit run app.py
"""

import streamlit as st
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.parser.parser import parse, ParseError
from src.analyzer.unified_analyzer import analyze_complexity_unified
from src.visualization.components import (
    display_procedure_analysis,  # NUEVO
)

# NUEVO: Importar sistema LLM
try:
    from src.llm.unified_analyzer_llm import analyze_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Analizador de Complejidad",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# EJEMPLOS PRECARGADOS (sin cambios)
# ============================================================================

EXAMPLES = {
    "🔹 Bubble Sort (Iterativo)": """BubbleSort(A[], n)
begin
    for i ← 1 to n-1 do
    begin
        for j ← 1 to n-i do
        begin
            if (A[j] > A[j+1]) then
            begin
                temp ← A[j]
                A[j] ← A[j+1]
                A[j+1] ← temp
            end
        end
    end
end""",
    
    "🔸 Merge Sort (Recursivo)": """MergeSort(A[], p, r)
begin
    if (p < r) then
    begin
        q ← floor((p + r) / 2)
        call MergeSort(A, p, q)
        call MergeSort(A, q+1, r)
        call Merge(A, p, q, r)
    end
end""",
    
    "⚡ Quick Sort (Híbrido)": """QuickSort(A[], p, r)
begin
    if (p < r) then
    begin
        q ← call Partition(A, p, r)
        call QuickSort(A, p, q-1)
        call QuickSort(A, q+1, r)
    end
end

Partition(A[], p, r)
begin
    pivot ← A[r]
    i ← p - 1
    
    for j ← p to r-1 do
    begin
        if (A[j] ≤ pivot) then
        begin
            i ← i + 1
            temp ← A[i]
            A[i] ← A[j]
            A[j] ← temp
        end
    end
    
    return i+1
end""",
    
    "🔍 Binary Search (Recursivo)": """BinarySearch(A[], left, right, x)
begin
    if (left > right) then
    begin
        return -1
    end
    
    mid ← floor((left + right) / 2)
    
    if (A[mid] = x) then
    begin
        return mid
    end
    
    if (A[mid] < x) then
    begin
        return call BinarySearch(A, mid+1, right, x)
    end
    else
    begin
        return call BinarySearch(A, left, mid-1, x)
    end
end""",
    
    "🔢 Factorial (Recursivo Lineal)": """Factorial(n)
begin
    if (n ≤ 1) then
    begin
        return 1
    end
    else
    begin
        return n * call Factorial(n-1)
    end
end""",
    
    "🌀 Fibonacci (Recursivo Binario)": """Fibonacci(n)
begin
    if (n ≤ 1) then
    begin
        return n
    end
    else
    begin
        return call Fibonacci(n-1) + call Fibonacci(n-2)
    end
end"""
}


# ============================================================================
# HEADER
# ============================================================================

st.title("🔍 Analizador de Complejidad Computacional")
st.markdown("""
Analiza la complejidad de algoritmos escritos en pseudocódigo.
Soporta algoritmos **iterativos**, **recursivos** e **híbridos**.
""")

st.divider()


# ============================================================================
# SIDEBAR: EJEMPLOS + CONFIGURACIÓN LLM
# ============================================================================

with st.sidebar:
    st.header("📚 Ejemplos")
    st.markdown("Selecciona un ejemplo para cargar automáticamente:")
    
    selected_example = st.selectbox(
        "Algoritmo:",
        options=[""] + list(EXAMPLES.keys()),
        format_func=lambda x: "-- Seleccionar --" if x == "" else x
    )
    
    if selected_example and selected_example in EXAMPLES:
        if st.button("📥 Cargar Ejemplo", use_container_width=True):
            st.session_state['code_input'] = EXAMPLES[selected_example]
            st.rerun()
    
    st.divider()
    
    # ========================================================================
    # NUEVO: CONFIGURACIÓN LLM
    # ========================================================================
    
    st.header("🤖 Validación con IA")
    
    if LLM_AVAILABLE:
        enable_llm = st.toggle(
            "Habilitar validación con LLM",
            value=False,
            help="Compara tu análisis con Llama 3.3 70B (Groq API)"
        )
        
        if enable_llm:
            st.info("💡 **Validación con IA habilitada**")
            
            # Verificar si hay API key en variable de entorno
            api_key_env = os.getenv('GROQ_API_KEY')
            
            if api_key_env:
                st.success("✓ API Key detectada en variables de entorno")
                api_key = api_key_env
            else:
                st.warning("⚠️ No se detectó GROQ_API_KEY en variables de entorno")
                api_key = st.text_input(
                    "API Key de Groq:",
                    type="password",
                    help="Obtén tu API key gratuita en https://console.groq.com/keys"
                )
                
                if not api_key:
                    st.error("❌ Ingresa tu API key para usar validación LLM")
            
            # Guardar en session state
            if api_key:
                st.session_state['groq_api_key'] = api_key
                st.session_state['llm_enabled'] = True
            else:
                st.session_state['llm_enabled'] = False
        else:
            st.session_state['llm_enabled'] = False
            st.info("ℹ️ Validación LLM deshabilitada")
    else:
        st.warning("⚠️ Módulo LLM no disponible")
        st.session_state['llm_enabled'] = False
    
    st.divider()
    
    # ========================================================================
    # SINTAXIS (sin cambios)
    # ========================================================================
    
    st.markdown("### ℹ️ Sintaxis")
    with st.expander("📖 Ver Guía Rápida"):
        st.markdown("""
**Ciclos:**
```
for i ← 1 to n do
while (condición) do
repeat ... until (condición)
```

**Condicionales:**
```
if (condición) then
begin
    ...
end
else
begin
    ...
end
```

**Recursión:**
```
call NombreProcedimiento(args)
return expresión
```

**Operadores:**
- Asignación: `←`
- Comparación: `<`, `>`, `≤`, `≥`, `=`, `≠`
- Aritméticos: `+`, `-`, `*`, `/`, `mod`, `div`, `^`
- Lógicos: `and`, `or`, `not`
""")


# ============================================================================
# MAIN: ENTRADA DE CÓDIGO (sin cambios)
# ============================================================================

st.header("✏️ Entrada de Código")

code_input = st.text_area(
    "Escribe o pega tu pseudocódigo:",
    value=st.session_state.get('code_input', ''),
    height=300,
    key='code_area'
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    analyze_button = st.button("🚀 Analizar", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Limpiar", use_container_width=True)

if clear_button:
    st.session_state['code_input'] = ''
    st.rerun()


# ============================================================================
# ANÁLISIS Y RESULTADOS (MEJORADO CON LLM)
# ============================================================================

if analyze_button:
    if not code_input.strip():
        st.error("⚠️ Por favor, ingresa código pseudocódigo para analizar.")
    else:
        # Verificar si LLM está habilitado
        llm_enabled = st.session_state.get('llm_enabled', False)
        
        with st.spinner("🔄 Analizando algoritmo..."):
            try:
                # ============================================================
                # SIN LLM: Análisis normal
                # ============================================================
                
                if not llm_enabled:
                    # Paso 1: Parsear
                    ast = parse(code_input)
                    
                    # Paso 2: Analizar
                    results = analyze_complexity_unified(ast)
                    
                    # Paso 3: Mostrar
                    st.success("✅ Análisis completado exitosamente")
                    st.divider()
                    st.header("📊 Resultados del Análisis")
                    
                    display_procedure_analysis(results)                    
                
                # ============================================================
                # CON LLM: Análisis + Validación
                # ============================================================
                
                else:
                    # Verificar API key
                    api_key = st.session_state.get('groq_api_key')
                    
                    if not api_key:
                        st.error("❌ API key no configurada. Desactiva LLM o configura tu API key.")
                    else:
                        # Configurar API key como variable de entorno temporal
                        os.environ['GROQ_API_KEY'] = api_key
                        
                        # Analizar con LLM
                        results = analyze_with_llm(code_input, enable_llm=True)
                        
                        st.success("✅ Análisis completado (con validación IA)")
                        
                        st.divider()
                        st.header("��� Resultados del Análisis con IA")
                        display_procedure_analysis(results)
                                
            except ParseError as e:
                st.error(f"❌ **Error de Sintaxis**")
                st.code(str(e), language=None)
                st.info("💡 Verifica que el código siga la sintaxis correcta del pseudocódigo.")
                
            except Exception as e:
                st.error(f"❌ **Error Inesperado**")
                st.code(str(e), language=None)
                
                # Mostrar traceback en expander para debugging
                with st.expander("🔍 Ver detalles del error"):
                    import traceback
                    st.code(traceback.format_exc())


# ============================================================================
# FOOTER (sin cambios)
# ============================================================================

st.divider()

st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
    <p>
        🎓 Proyecto de Análisis y Diseño de Algoritmos<br>
        Universidad: Universidad de Caldas | 2025
    </p>
</div>
""", unsafe_allow_html=True)