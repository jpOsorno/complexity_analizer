"""
Componentes de Visualización
============================

Funciones helper para mostrar resultados del análisis en Streamlit.
"""

import streamlit as st
from typing import Dict, Optional
import json


def create_complexity_badge(complexity: str, case_type: str) -> str:
    """
    Crea un badge HTML para mostrar complejidad.
    
    Args:
        complexity: Complejidad (ej: "O(n²)")
        case_type: Tipo de caso ("worst", "best", "average")
    
    Returns:
        HTML string con el badge estilizado
    """
    # Colores según el tipo de caso
    colors = {
        "worst": "#ef4444",    # Rojo
        "best": "#10b981",     # Verde
        "average": "#f59e0b"   # Amarillo
    }
    
    color = colors.get(case_type, "#6b7280")
    
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.375rem;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
        margin: 0.25rem;
    ">
        {complexity}
    </span>
    """


def format_equation(equation: str) -> str:
    """
    Formatea ecuación de recurrencia para mejor visualización.
    
    Args:
        equation: Ecuación (ej: "T(n) = 2T(n/2) + O(n)")
    
    Returns:
        Ecuación formateada con HTML/Markdown
    """
    if not equation:
        return "*No disponible*"
    
    # Reemplazar símbolos matemáticos
    formatted = equation.replace("O(", "**O(**").replace(")", "**)**")
    formatted = formatted.replace("Θ(", "**Θ(**")
    formatted = formatted.replace("Ω(", "**Ω(**")
    formatted = formatted.replace("T(", "**T(**")
    
    return formatted


def display_complexity_result(result, procedure_name: str):
    """
    Muestra el resultado del análisis de complejidad de un procedimiento.
    
    Args:
        result: UnifiedComplexityResult o ComplexityResult
        procedure_name: Nombre del procedimiento
    """
    st.subheader(f"📊 {procedure_name}")
    
    # Tipo de algoritmo
    algo_type = getattr(result, 'algorithm_type', 'iterative')
    type_emoji = {
        'iterative': '🔄',
        'recursive': '🔁',
        'hybrid': '⚡'
    }
    
    st.markdown(f"**Tipo:** {type_emoji.get(algo_type, '📝')} {algo_type.title()}")
    
    # Complejidades principales
    st.markdown("### Complejidad Computacional")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Peor Caso**")
        worst = getattr(result, 'final_worst', None) or getattr(result, 'worst_case', 'O(?)')
        st.markdown(create_complexity_badge(worst, "worst"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Mejor Caso**")
        best = getattr(result, 'final_best', None) or getattr(result, 'best_case', 'Ω(?)')
        st.markdown(create_complexity_badge(best, "best"), unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Caso Promedio**")
        avg = getattr(result, 'final_average', None) or getattr(result, 'average_case', 'Θ(?)')
        st.markdown(create_complexity_badge(avg, "average"), unsafe_allow_html=True)
    
    # Análisis recursivo (si aplica)
    if getattr(result, 'is_recursive', False):
        st.markdown("### 🔁 Análisis Recursivo")
        
        rec_eq = getattr(result, 'recurrence_equation', None)
        if rec_eq:
            st.markdown(f"**Ecuación:** `{rec_eq}`")
            
            # Solución de recurrencia
            rec_sol = getattr(result, 'recurrence_solution', None)
            if rec_sol:
                st.markdown(f"**Solución:** {rec_sol.big_theta}")
                st.markdown(f"**Método:** {rec_sol.method_used}")
    
    # Explicación
    explanation = getattr(result, 'explanation', '')
    if explanation:
        with st.expander("📝 Explicación Detallada"):
            st.markdown(explanation)
    
    # Pasos del análisis
    steps = getattr(result, 'steps', [])
    if steps:
        with st.expander("🔍 Pasos del Análisis"):
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")
    
    st.divider()


def display_procedure_analysis(results: Dict):
    """
    Muestra resultados de múltiples procedimientos en tabs.
    
    Args:
        results: Dict con resultados por procedimiento
    """
    if not results:
        st.warning("⚠️ No se encontraron procedimientos para analizar.")
        return
    
    # Si hay un solo procedimiento, mostrarlo directo
    if len(results) == 1:
        proc_name, result = next(iter(results.items()))
        display_complexity_result(result, proc_name)
        return
    
    # Si hay múltiples procedimientos, usar tabs
    proc_names = list(results.keys())
    tabs = st.tabs(proc_names)
    
    for tab, proc_name in zip(tabs, proc_names):
        with tab:
            display_complexity_result(results[proc_name], proc_name)


def export_results_json(results: Dict) -> str:
    """
    Exporta resultados a JSON.
    
    Args:
        results: Dict con resultados del análisis
    
    Returns:
        JSON string
    """
    export_data = {}
    
    for proc_name, result in results.items():
        # Intentar usar to_dict() si existe
        if hasattr(result, 'to_dict'):
            export_data[proc_name] = result.to_dict()
        else:
            # Fallback: extraer atributos manualmente
            export_data[proc_name] = {
                "worst_case": getattr(result, 'final_worst', None) or getattr(result, 'worst_case', 'O(?)'),
                "best_case": getattr(result, 'final_best', None) or getattr(result, 'best_case', 'Ω(?)'),
                "average_case": getattr(result, 'final_average', None) or getattr(result, 'average_case', 'Θ(?)'),
                "algorithm_type": getattr(result, 'algorithm_type', 'unknown'),
                "is_recursive": getattr(result, 'is_recursive', False),
                "explanation": getattr(result, 'explanation', '')
            }
    
    return json.dumps(export_data, indent=2)