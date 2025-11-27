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
    
    # ========================================================================
    # ANÁLISIS ITERATIVO DETALLADO - NUEVO
    # ========================================================================
    
    iterative_analysis = getattr(result, 'iterative_analysis', None)
    
    if iterative_analysis and algo_type == "iterative":
        st.markdown("### 🔄 Análisis Iterativo Detallado")
        
        # Crear tabs para cada caso
        tab_worst, tab_best, tab_avg = st.tabs([
            "🔴 Peor Caso",
            "🟢 Mejor Caso",
            "🟡 Caso Promedio"
        ])
        
        # ============================================================
        # TAB: PEOR CASO
        # ============================================================
        with tab_worst:
            st.markdown("#### Sumatoria / Ecuación")
            if iterative_analysis.worst_case_summation:
                st.code(iterative_analysis.worst_case_summation, language=None)
            else:
                st.info("No hay sumatorias (complejidad constante)")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Complejidad:**")
                st.error(iterative_analysis.worst_case)
            
            with col_b:
                st.markdown("**Estructura:**")
                st.info(f"{len(iterative_analysis.loops)} ciclo(s), profundidad {iterative_analysis.max_nesting_depth}")
            
            # Explicación
            if iterative_analysis.worst_case_explanation:
                st.markdown("**Explicación:**")
                st.markdown(iterative_analysis.worst_case_explanation)
            
            # Pasos del análisis
            if iterative_analysis.worst_case_steps:
                st.markdown("**Pasos del Análisis:**")
                for i, step in enumerate(iterative_analysis.worst_case_steps, 1):
                    st.markdown(f"{i}. {step}")
        
        # ============================================================
        # TAB: MEJOR CASO
        # ============================================================
        with tab_best:
            st.markdown("#### Sumatoria / Ecuación")
            if iterative_analysis.best_case_summation:
                st.code(iterative_analysis.best_case_summation, language=None)
            else:
                st.info("No hay sumatorias (complejidad constante)")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Complejidad:**")
                st.success(iterative_analysis.best_case)
            
            with col_b:
                st.markdown("**Early Exit:**")
                if iterative_analysis.has_early_exit:
                    st.success("✓ Detectado")
                else:
                    st.info("No detectado")
            
            # Explicación
            if iterative_analysis.best_case_explanation:
                st.markdown("**Explicación:**")
                st.markdown(iterative_analysis.best_case_explanation)
            
            # Pasos
            if iterative_analysis.best_case_steps:
                st.markdown("**Pasos del Análisis:**")
                for i, step in enumerate(iterative_analysis.best_case_steps, 1):
                    st.markdown(f"{i}. {step}")
        
        # ============================================================
        # TAB: CASO PROMEDIO
        # ============================================================
        with tab_avg:
            st.markdown("#### Sumatoria / Ecuación")
            if iterative_analysis.average_case_summation:
                st.code(iterative_analysis.average_case_summation, language=None)
            else:
                st.info("No hay sumatorias (complejidad constante)")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Complejidad:**")
                st.warning(iterative_analysis.average_case)
            
            with col_b:
                st.markdown("**Condicionales:**")
                if iterative_analysis.has_conditionals:
                    st.info("✓ Detectados")
                else:
                    st.info("No detectados")
            
            # Explicación
            if iterative_analysis.average_case_explanation:
                st.markdown("**Explicación:**")
                st.markdown(iterative_analysis.average_case_explanation)
            
            # Pasos
            if iterative_analysis.average_case_steps:
                st.markdown("**Pasos del Análisis:**")
                for i, step in enumerate(iterative_analysis.average_case_steps, 1):
                    st.markdown(f"{i}. {step}")

    # ========================================================================
    # ANÁLISIS RECURSIVO DETALLADO - MEJORADO
    # ========================================================================

    if getattr(result, 'is_recursive', False):
        st.markdown("### 🔁 Análisis Recursivo Detallado")
        
        recurrence_analysis = getattr(result, 'recurrence_analysis', None)
        
        if recurrence_analysis:
            # Crear tabs para cada caso
            tab_worst, tab_best, tab_avg = st.tabs([
                "🔴 Peor Caso", 
                "🟢 Mejor Caso", 
                "🟡 Caso Promedio"
            ])
            
            # ============================================================
            # TAB: PEOR CASO
            # ============================================================
            with tab_worst:
                st.markdown("#### Ecuación de Recurrencia")
                st.code(recurrence_analysis.worst_case_equation, language=None)
                
                if recurrence_analysis.worst_case_solution:
                    sol = recurrence_analysis.worst_case_solution
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Solución:**")
                        st.info(sol.big_theta)
                    
                    with col_b:
                        st.markdown("**Método:**")
                        st.info(sol.method_used)
                    
                    # Explicación
                    if recurrence_analysis.worst_case_explanation:
                        st.markdown("**Explicación:**")
                        st.markdown(recurrence_analysis.worst_case_explanation)
                    
                    # Pasos de resolución
                    if sol.steps:
                        st.markdown("**Pasos de Resolución:**")
                        for i, step in enumerate(sol.steps, 1):
                            st.markdown(f"{i}. {step}")
            
            # ============================================================
            # TAB: MEJOR CASO
            # ============================================================
            with tab_best:
                st.markdown("#### Ecuación de Recurrencia")
                st.code(recurrence_analysis.best_case_equation, language=None)
                
                if recurrence_analysis.best_case_solution:
                    sol = recurrence_analysis.best_case_solution
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Solución:**")
                        st.success(sol.big_theta)
                    
                    with col_b:
                        st.markdown("**Método:**")
                        st.info(sol.method_used)
                    
                    # Explicación
                    if recurrence_analysis.best_case_explanation:
                        st.markdown("**Explicación:**")
                        st.markdown(recurrence_analysis.best_case_explanation)
                    
                    # Pasos de resolución
                    if sol.steps:
                        st.markdown("**Pasos de Resolución:**")
                        for i, step in enumerate(sol.steps, 1):
                            st.markdown(f"{i}. {step}")
            
            # ============================================================
            # TAB: CASO PROMEDIO - NUEVO
            # ============================================================
            with tab_avg:
                st.markdown("#### Ecuación de Recurrencia")
                st.code(recurrence_analysis.average_case_equation, language=None)
                
                if recurrence_analysis.average_case_solution:
                    sol = recurrence_analysis.average_case_solution
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Solución:**")
                        st.warning(sol.big_theta)
                    
                    with col_b:
                        st.markdown("**Método:**")
                        st.info(sol.method_used)
                    
                    # Explicación
                    if recurrence_analysis.average_case_explanation:
                        st.markdown("**Explicación:**")
                        st.markdown(recurrence_analysis.average_case_explanation)
                    
                    # Pasos de resolución
                    if sol.steps:
                        st.markdown("**Pasos de Resolución:**")
                        for i, step in enumerate(sol.steps, 1):
                            st.markdown(f"{i}. {step}")

    # Explicación general (ahora con los 3 casos incluidos)
    explanation = getattr(result, 'explanation', '')
    if explanation:
        with st.expander("📝 Explicación Completa del Análisis"):
            st.markdown(explanation)

    # Pasos del análisis general
    steps = getattr(result, 'steps', [])
    if steps:
        with st.expander("🔍 Pasos del Análisis General"):
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