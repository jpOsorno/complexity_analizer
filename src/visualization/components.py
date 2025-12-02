"""
Componentes de Visualización
============================

Funciones helper para mostrar resultados del análisis en Streamlit.
"""

import streamlit as st
from typing import Dict, Optional
import numpy as np
import json
import hashlib
import uuid
from .complexity_plotter import ComplexityPlotter, plot_complexity

# NUEVO: Import del visualizador de árboles
try:
    from .recursion_tree_visualizer import (
        RecursionTreeVisualizer, 
        RecursionTreeBuilder,
        visualize_divide_conquer_tree,
        visualize_fibonacci_tree
    )
    TREE_VISUALIZATION_AVAILABLE = True
except ImportError:
    TREE_VISUALIZATION_AVAILABLE = False


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


# ============================================================================
# NUEVA FUNCIÓN: GRÁFICOS DE COMPLEJIDAD
# ============================================================================

def display_complexity_plots(
    worst: str,
    best: str,
    average: str,
    procedure_name: str
):
    """
    Muestra gráficos interactivos de las funciones de complejidad.
    
    Args:
        worst: Complejidad del peor caso
        best: Complejidad del mejor caso
        average: Complejidad del caso promedio
        procedure_name: Nombre del procedimiento
    """
    st.markdown("### 📈 Visualización de Complejidades")
    
    try:
        # Crear graficador
        plotter = ComplexityPlotter(theme="plotly_white")
        
        # Generar gráfico
        fig = plotter.plot_three_cases(
            worst=worst,
            best=best,
            average=average,
            title=f"Análisis de Complejidad: {procedure_name}"
        )
        
        # Mostrar en Streamlit (usar key única para evitar StreamlitDuplicateElementId)
        try:
            base = hashlib.md5(f"{procedure_name}_{worst}_{best}_{average}".encode()).hexdigest()[:8]
            key = f"complexity_plot_{base}_{uuid.uuid4().hex[:6]}"
        except Exception:
            key = f"complexity_plot_{uuid.uuid4().hex[:8]}"
        st.plotly_chart(fig, use_container_width=True, key=key)
        
        # Información adicional
        with st.expander("ℹ️ Cómo interpretar este gráfico"):
            st.markdown("""
**Eje X (horizontal):** Tamaño de la entrada (n)
- Representa cuántos elementos tiene el problema a resolver
- Ejemplo: para ordenar un array, n es el número de elementos

**Eje Y (vertical):** Número de operaciones T(n)
- Representa cuántas operaciones básicas ejecuta el algoritmo
- A mayor altura, más tiempo de ejecución

**Líneas de colores:**
- 🔴 **Rojo (Peor Caso):** Máximo número de operaciones posibles
- 🟢 **Verde (Mejor Caso):** Mínimo número de operaciones posibles
- 🟡 **Amarillo (Caso Promedio):** Comportamiento típico esperado

**Escala Logarítmica:**
- Si el eje Y usa escala logarítmica, es porque las complejidades tienen rangos muy diferentes
- Útil para comparar complejidades exponenciales vs polinómicas
            """)
    
    except Exception as e:
        st.warning(f"⚠️ No se pudo generar el gráfico: {e}")
        st.info("El análisis de complejidad está disponible en formato texto arriba.")


# ============================================================================
# NUEVA FUNCIÓN: VISUALIZACIÓN DE ÁRBOLES DE RECURSIÓN
# ============================================================================

# ============================================================================
# CORRECCIÓN: VISUALIZACIÓN DE ÁRBOLES DE RECURSIÓN
# ============================================================================

def display_recursion_tree_visualization(recurrence_equation: str, method_used: str):
    """
    Muestra visualización del árbol de recursión si es aplicable.
    
    Args:
        recurrence_equation: Ecuación de recurrencia
        method_used: Método usado para resolver
    """
    if not TREE_VISUALIZATION_AVAILABLE:
        st.warning("⚠️ Módulo de visualización de árboles no disponible")
        st.info("Instala las dependencias: `pip install plotly`")
        return
    
    st.markdown("#### 🌳 Visualización del Árbol de Recursión")
    
    # Mostrar ecuación que estamos analizando
    st.info(f"**Ecuación:** `{recurrence_equation}`")
    
    try:
        import re
        
        # Limpiar ecuación
        eq_clean = recurrence_equation.replace(" ", "").replace("T(n)=", "")
        
        # # Debug: mostrar ecuación limpia
        # with st.expander("🔍 Debug: Ver ecuación procesada"):
        #     st.code(f"Original: {recurrence_equation}\nLimpia: {eq_clean}")
        
        tree_generated = False
        
        # ====================================================================
        # PATRÓN 1: Divide y Vencerás - T(n) = aT(n/b) + f(n)
        # ====================================================================
        
        # Intentar múltiples patrones
        patterns = [
            r'(\d+)T\(n/(\d+)\)',           # 2T(n/2)
            r'(\d+)\*?T\(n/(\d+)\)',        # 2*T(n/2)
            r'T\(n/(\d+)\)',                # T(n/2) - asume a=1
        ]
        
        for pattern in patterns:
            match = re.search(pattern, eq_clean)
            if match:
                if len(match.groups()) == 2:
                    a = int(match.group(1))
                    b = int(match.group(2))
                elif len(match.groups()) == 1:
                    a = 1
                    b = int(match.group(1))
                else:
                    continue
                
                # Extraer f(n)
                if "O(n^2)" in eq_clean or "O(n²)" in eq_clean:
                    f_n = "O(n²)"
                elif "O(n)" in eq_clean or "+n" in eq_clean:
                    f_n = "O(n)"
                elif "O(logn)" in eq_clean or "O(log(n))" in eq_clean:
                    f_n = "O(log n)"
                elif "O(1)" in eq_clean or "+c" in eq_clean or "+1" in eq_clean:
                    f_n = "O(1)"
                else:
                    f_n = "O(1)"
                
                st.success(f"✓ Detectado: Divide y Vencerás → a={a}, b={b}, f(n)={f_n}")
                
                # Determinar profundidad basada en a y b
                if a >= 4 or b >= 3:
                    max_depth = 3
                elif a == 1:
                    max_depth = 6
                else:
                    max_depth = 4
                
                with st.spinner("Generando árbol..."):
                    fig = visualize_divide_conquer_tree(a, b, f_n, max_depth=max_depth)
                    try:
                        base = hashlib.md5(f"{recurrence_equation}_div_{a}_{b}_{f_n}_{max_depth}".encode()).hexdigest()[:8]
                        key = f"rec_tree_{base}_{uuid.uuid4().hex[:6]}"
                    except Exception:
                        key = f"rec_tree_{uuid.uuid4().hex[:8]}"
                    st.plotly_chart(fig, use_container_width=True, key=key)
                
                # Información adicional
                with st.expander("ℹ️ Cómo interpretar el árbol"):
                    st.markdown(f"""
**Estructura del árbol:**
- **Ramificación:** Cada nodo tiene {a} hijo(s)
- **Factor de división:** Cada nivel divide por {b}
- **Altura del árbol:** log_{b}(n) niveles
- **Número de hojas:** {a}^log_{b}(n) = n^log_{b}({a}) ≈ n^{round(np.log(a)/np.log(b), 2)}

**Colores:**
- Los colores indican el nivel de profundidad
- Azul = raíz, verde/amarillo = niveles medios, rojo = hojas

**Costo por nivel:**
- Nivel 0: 1 nodo → {f_n}
- Nivel 1: {a} nodos → {a}×{f_n}
- Nivel k: {a}^k nodos → {a}^k×{f_n}

**Interactividad:**
- Haz hover sobre los nodos para ver detalles
- Usa zoom y pan para explorar
                    """)
                
                tree_generated = True
                break
        
        # ====================================================================
        # PATRÓN 2: Fibonacci - T(n) = T(n-1) + T(n-2)
        # ====================================================================
        
        if not tree_generated:
            if "T(n-1)" in eq_clean and "T(n-2)" in eq_clean:
                st.success("✓ Detectado: Fibonacci (recursión binaria)")
                
                with st.spinner("Generando árbol de Fibonacci..."):
                    fig = visualize_fibonacci_tree(max_depth=5)
                    try:
                        base = hashlib.md5(f"{recurrence_equation}_fib_5".encode()).hexdigest()[:8]
                        key = f"rec_tree_{base}_{uuid.uuid4().hex[:6]}"
                    except Exception:
                        key = f"rec_tree_{uuid.uuid4().hex[:8]}"
                    st.plotly_chart(fig, use_container_width=True, key=key)
                
                with st.expander("ℹ️ Cómo interpretar el árbol de Fibonacci"):
                    st.markdown("""
**Estructura:**
- Cada nodo representa F(k) para algún k
- Cada nodo tiene exactamente 2 hijos: F(k-1) y F(k-2)
- El árbol crece exponencialmente

**Colores:**
- Diferentes colores por nivel de profundidad
- Ayuda a visualizar la expansión exponencial

**Problema de eficiencia:**
- **Muchos cálculos duplicados** (nodos repetidos)
- F(n-3) se calcula 3 veces
- F(n-4) se calcula 5 veces
- Esto causa la complejidad O(φⁿ) ≈ O(1.618ⁿ)

**Mejora posible:**
- Programación dinámica (memoización)
- Reduciría complejidad a O(n)
- Tabla de valores precalculados

**Número áureo (φ):**
- φ = (1 + √5) / 2 ≈ 1.618
- Fibonacci(n) ≈ φⁿ / √5
                    """)
                
                tree_generated = True
        
        # ====================================================================
        # PATRÓN 3: Recursión Lineal - T(n) = T(n-k) + f(n)
        # ====================================================================
        
        if not tree_generated:
            linear_match = re.search(r'T\(n-(\d+)\)', eq_clean)
            
            if linear_match:
                k = int(linear_match.group(1))
                
                if "O(n)" in eq_clean:
                    f_n = "O(n)"
                else:
                    f_n = "O(1)"
                
                st.success(f"✓ Detectado: Recursión Lineal → k={k}, f(n)={f_n}")
                
                # Para recursión lineal, mostrar diagrama de cadena
                st.markdown("**Estructura: Cadena Lineal (no ramifica)**")
                
                # Crear visualización simple de cadena
                chain_viz = "```\n"
                chain_viz += "T(n) → T(n-{}) → T(n-{}) → ... → T(1) → T(0)\n".format(k, 2*k)
                chain_viz += "  ↓       ↓         ↓               ↓       ↓\n"
                chain_viz += " {}      {}        {}              {}      {}\n".format(f_n, f_n, f_n, f_n, "base")
                chain_viz += "\n"
                chain_viz += "Profundidad: n/{} niveles\n".format(k)
                chain_viz += "Costo total: (n/{}) × {} = Θ(n)\n".format(k, f_n)
                chain_viz += "```"
                
                st.markdown(chain_viz)
                
                with st.expander("ℹ️ Explicación"):
                    st.markdown(f"""
**Características:**
- No hay ramificación (un solo hijo por nodo)
- Es una cadena lineal de llamadas
- Profundidad: n/{k} niveles
- Cada nivel tiene costo {f_n}

**Complejidad:**
- Si f(n) = O(1): Total = (n/{k}) × O(1) = O(n)
- Si f(n) = O(n): Total = (n/{k}) × O(n) = O(n²)

**Comparación con árboles:**
- Árboles como Fibonacci ramifican → crecimiento exponencial
- Cadenas lineales no ramifican → crecimiento lineal
                    """)
                
                tree_generated = True
        
        # ====================================================================
        # NO SE PUDO GENERAR
        # ====================================================================
        
        if not tree_generated:
            st.warning("⚠️ No se pudo generar visualización automática del árbol")
            
            st.info("""
**Patrones soportados:**
1. **Divide y Vencerás:** `T(n) = aT(n/b) + f(n)`
   - Ejemplo: `T(n) = 2T(n/2) + O(n)` (Merge Sort)
   - Ejemplo: `T(n) = 7T(n/2) + O(n²)` (Strassen)

2. **Fibonacci:** `T(n) = T(n-1) + T(n-2) + O(1)`

3. **Recursión Lineal:** `T(n) = T(n-k) + f(n)`
   - Ejemplo: `T(n) = T(n-1) + O(1)` (Factorial)

**Tu ecuación:** `{}`

Si tu ecuación sigue uno de estos patrones pero no se detectó,
por favor reporta el problema.
            """.format(recurrence_equation))
            
            # Mostrar análisis textual como fallback
            st.markdown("**Análisis Textual del Árbol:**")
            
            # Intentar extraer del tree_analysis si existe
            if method_used and "tree" in method_used.lower():
                st.info("Ver la sección 'Explicación detallada' en los tabs de casos para el análisis del árbol")
    
    except Exception as e:
        st.error(f"❌ Error al generar visualización: {e}")
        
        # Debug completo
        with st.expander("🐛 Ver error completo (para debugging)"):
            import traceback
            st.code(traceback.format_exc())
        
        st.info("""
**Problema detectado al generar el árbol.**

Posibles causas:
1. La ecuación tiene un formato no estándar
2. Falta alguna dependencia (plotly)
3. Error en el parsing

**Solución temporal:**
- Revisa la explicación textual en los tabs de casos
- El análisis de complejidad sigue siendo correcto
        """)

# ============================================================================
# NUEVA FUNCIÓN: RENDERIZAR PASOS CON FORMATO MEJORADO
# ============================================================================

def display_solution_steps(steps: list, title: str = "Pasos de Resolución"):
    """
    Muestra los pasos de resolución con formato mejorado.
    
    Args:
        steps: Lista de strings con los pasos
        title: Título de la sección
    """
    if not steps:
        return
    
    st.markdown(f"#### {title}")
    
    # Detectar si hay tablas en los pasos
    has_tables = any("╔" in step or "║" in step for step in steps)
    
    if has_tables:
        # Mostrar en expander con fuente monoespaciada
        with st.expander("📋 Ver pasos detallados"):
            for step in steps:
                if "╔" in step or "║" in step or "╠" in step or "╚" in step:
                    # Es una tabla, mostrar con código
                    st.code(step, language=None)
                elif step.strip() == "":
                    # Línea vacía
                    st.markdown("")
                elif step.startswith("🔍") or step.startswith("📐") or step.startswith("✅"):
                    # Título de sección
                    st.markdown(f"**{step}**")
                elif step.startswith("  •") or step.startswith("    •"):
                    # Item de lista
                    st.markdown(step)
                elif step.startswith("💡"):
                    # Observación especial
                    st.info(step)
                else:
                    # Texto normal
                    st.markdown(step)
    else:
        # Sin tablas, mostrar directo
        for step in steps:
            if step.strip() == "":
                st.markdown("")
            elif step.startswith("🔍") or step.startswith("📐") or step.startswith("✅"):
                st.markdown(f"**{step}**")
            elif step.startswith("💡"):
                st.info(step)
            else:
                st.markdown(step)


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
    # GRÁFICOS DE COMPLEJIDAD
    # ========================================================================
    
    display_complexity_plots(worst, best, avg, procedure_name)
    
    # ========================================================================
    # ANÁLISIS ITERATIVO DETALLADO
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
        
        # TAB: PEOR CASO
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
            
            with st.expander("🔎 Explicación detallada"):
                if iterative_analysis.worst_case_explanation:
                    st.markdown("**Explicación:**")
                    st.markdown(iterative_analysis.worst_case_explanation)
                
                # MEJORADO: Pasos con formato
                if iterative_analysis.worst_case_steps:
                    display_solution_steps(iterative_analysis.worst_case_steps, "Pasos del Análisis")
        
        # TAB: MEJOR CASO
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
            
            with st.expander("🔎 Explicación detallada"):
                if iterative_analysis.best_case_explanation:
                    st.markdown("**Explicación:**")
                    st.markdown(iterative_analysis.best_case_explanation)
                
                if iterative_analysis.best_case_steps:
                    display_solution_steps(iterative_analysis.best_case_steps, "Pasos del Análisis")
        
        # TAB: CASO PROMEDIO
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
            
            with st.expander("🔎 Explicación detallada"):
                if iterative_analysis.average_case_explanation:
                    st.markdown("**Explicación:**")
                    st.markdown(iterative_analysis.average_case_explanation)
                
                if iterative_analysis.average_case_steps:
                    display_solution_steps(iterative_analysis.average_case_steps, "Pasos del Análisis")

    # ========================================================================
    # ANÁLISIS RECURSIVO DETALLADO - MEJORADO CON VISUALIZACIONES
    # ========================================================================

    if getattr(result, 'is_recursive', False):
        st.markdown("### 🔁 Análisis Recursivo Detallado")
        
        recurrence_analysis = getattr(result, 'recurrence_analysis', None)
        
        if recurrence_analysis:
            # Crear tabs para cada caso
            tab_worst, tab_best, tab_avg= st.tabs([
                "🔴 Peor Caso", 
                "🟢 Mejor Caso", 
                "🟡 Caso Promedio"
            ])
            
            # TAB: PEOR CASO
            with tab_worst:
                st.markdown("#### Ecuación de Recurrencia")
                st.code(recurrence_analysis.worst_case_equation, language=None)
                
                if recurrence_analysis.worst_case_solution:
                    sol = recurrence_analysis.worst_case_solution
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown("**Solución:**")
                        st.error(sol.big_theta)
                    
                    with col_b:
                        st.markdown("**Método:**")
                        st.info(sol.method_used)
                    
                    with col_c:
                        st.markdown("**Clase:**")
                        st.info(sol.complexity_class)
                    
                    # Cotas fuertes
                    if sol.tight_bounds:
                        st.markdown("**Cotas Fuertes (Tight Bounds):**")
                        st.latex(sol.tight_bounds.replace("≤", r"\leq").replace("₁", "_1").replace("₂", "_2"))
                    
                    with st.expander("🔎 Explicación detallada"):
                        if recurrence_analysis.worst_case_explanation:
                            st.markdown("**Explicación:**")
                            st.markdown(recurrence_analysis.worst_case_explanation)
                        
                        # MEJORADO: Pasos con formato
                        if sol.steps:
                            display_solution_steps(sol.steps, "Pasos de Resolución")
                        
                        # Análisis del árbol (si existe)
                        if sol.tree_analysis:
                            st.markdown("---")
                            st.markdown("**Análisis del Árbol:**")
                            st.info(sol.tree_analysis)
                    
                    # NUEVO: Expander para gráficas y visualización del árbol
                    with st.expander("📈 Ver Gráficas y Árbol de Recursión"):
                        display_recursion_tree_visualization(
                            recurrence_analysis.worst_case_equation,
                            sol.method_used
                        )
            
            # TAB: MEJOR CASO
            with tab_best:
                st.markdown("#### Ecuación de Recurrencia")
                st.code(recurrence_analysis.best_case_equation, language=None)
                
                if recurrence_analysis.best_case_solution:
                    sol = recurrence_analysis.best_case_solution
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown("**Solución:**")
                        st.success(sol.big_theta)
                    
                    with col_b:
                        st.markdown("**Método:**")
                        st.info(sol.method_used)
                    
                    with col_c:
                        st.markdown("**Clase:**")
                        st.info(sol.complexity_class)
                    
                    if sol.tight_bounds:
                        st.markdown("**Cotas Fuertes:**")
                        st.latex(sol.tight_bounds.replace("≤", r"\leq").replace("₁", "_1").replace("₂", "_2"))
                    
                    with st.expander("🔎 Explicación detallada"):
                        if recurrence_analysis.best_case_explanation:
                            st.markdown("**Explicación:**")
                            st.markdown(recurrence_analysis.best_case_explanation)
                        
                        if sol.steps:
                            display_solution_steps(sol.steps, "Pasos de Resolución")
                        
                        if sol.tree_analysis:
                            st.markdown("---")
                            st.markdown("**Análisis del Árbol:**")
                            st.info(sol.tree_analysis)
                    
                    # NUEVO: Expander para gráficas y visualización del árbol
                    with st.expander("📈 Ver Gráficas y Árbol de Recursión"):
                        display_recursion_tree_visualization(
                            recurrence_analysis.best_case_equation,
                            sol.method_used
                        )
            
            # TAB: CASO PROMEDIO
            with tab_avg:
                st.markdown("#### Ecuación de Recurrencia")
                st.code(recurrence_analysis.average_case_equation, language=None)
                
                if recurrence_analysis.average_case_solution:
                    sol = recurrence_analysis.average_case_solution
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown("**Solución:**")
                        st.warning(sol.big_theta)
                    
                    with col_b:
                        st.markdown("**Método:**")
                        st.info(sol.method_used)
                    
                    with col_c:
                        st.markdown("**Clase:**")
                        st.info(sol.complexity_class)
                    
                    if sol.tight_bounds:
                        st.markdown("**Cotas Fuertes:**")
                        st.latex(sol.tight_bounds.replace("≤", r"\leq").replace("₁", "_1").replace("₂", "_2"))
                    
                    with st.expander("🔎 Explicación detallada"):
                        if recurrence_analysis.average_case_explanation:
                            st.markdown("**Explicación:**")
                            st.markdown(recurrence_analysis.average_case_explanation)
                        
                        if sol.steps:
                            display_solution_steps(sol.steps, "Pasos de Resolución")
                        
                        if sol.tree_analysis:
                            st.markdown("---")
                            st.markdown("**Análisis del Árbol:**")
                            st.info(sol.tree_analysis)
                    
                    # NUEVO: Expander para gráficas y visualización del árbol
                    with st.expander("📈 Ver Gráficas y Árbol de Recursión"):
                        display_recursion_tree_visualization(
                            recurrence_analysis.average_case_equation,
                            sol.method_used
                        )
            



# ... (mantener resto de funciones: display_procedure_analysis, display_llm_comparison, etc.) ...

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
        if hasattr(result, 'to_dict'):
            export_data[proc_name] = result.to_dict()
        else:
            export_data[proc_name] = {
                "worst_case": getattr(result, 'final_worst', None) or getattr(result, 'worst_case', 'O(?)'),
                "best_case": getattr(result, 'final_best', None) or getattr(result, 'best_case', 'Ω(?)'),
                "average_case": getattr(result, 'final_average', None) or getattr(result, 'average_case', 'Θ(?)'),
                "algorithm_type": getattr(result, 'algorithm_type', 'unknown'),
                "is_recursive": getattr(result, 'is_recursive', False),
                "explanation": getattr(result, 'explanation', '')
            }
    
    return json.dumps(export_data, indent=2)