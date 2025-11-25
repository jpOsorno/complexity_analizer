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
    display_procedure_analysis,
    export_results_json
)


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
# EJEMPLOS PRECARGADOS
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
# SIDEBAR: EJEMPLOS
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
# MAIN: ENTRADA DE CÓDIGO
# ============================================================================

st.header("✏️ Entrada de Código")

# Área de texto para el código
code_input = st.text_area(
    "Escribe o pega tu pseudocódigo:",
    value=st.session_state.get('code_input', ''),
    height=300,
    key='code_area'
)

# Botón de análisis
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    analyze_button = st.button("🚀 Analizar", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Limpiar", use_container_width=True)

if clear_button:
    st.session_state['code_input'] = ''
    st.rerun()


# ============================================================================
# ANÁLISIS Y RESULTADOS
# ============================================================================

if analyze_button:
    if not code_input.strip():
        st.error("⚠️ Por favor, ingresa código pseudocódigo para analizar.")
    else:
        with st.spinner("🔄 Analizando algoritmo..."):
            try:
                # Paso 1: Parsear
                ast = parse(code_input)
                
                # Paso 2: Analizar con sistema unificado
                results = analyze_complexity_unified(ast)
                
                # Paso 3: Mostrar resultados
                st.success("✅ Análisis completado exitosamente")
                
                st.divider()
                st.header("📊 Resultados del Análisis")
                
                # Mostrar resultados por procedimiento
                display_procedure_analysis(results)
                
                # Botón de descarga
                st.divider()
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    json_data = export_results_json(results)
                    st.download_button(
                        label="💾 Descargar JSON",
                        data=json_data,
                        file_name="analisis_complejidad.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
            except ParseError as e:
                st.error(f"❌ **Error de Sintaxis**")
                st.code(str(e), language=None)
                st.info("💡 Verifica que el código siga la sintaxis correcta del pseudocódigo.")
                
            except Exception as e:
                st.error(f"❌ **Error Inesperado**")
                st.code(str(e), language=None)
                st.warning("⚠️ Si el error persiste, contacta al desarrollador.")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
    <p>
        🎓 Proyecto de Análisis y Diseño de Algoritmos<br>
        Universidad: [Tu Universidad] | 2025
    </p>
</div>
""", unsafe_allow_html=True)