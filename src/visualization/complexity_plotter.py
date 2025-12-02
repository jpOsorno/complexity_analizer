"""
Graficador de Funciones de Complejidad
======================================

Genera gráficos interactivos de funciones O(n), Ω(n), Θ(n) usando Plotly.

Características:
- Múltiples complejidades en un solo gráfico
- Comparación visual de mejor, peor y caso promedio
- Interactividad con zoom y tooltips
- Escalas logarítmicas para complejidades exponenciales
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Tuple, Optional
import re


# ============================================================================
# FUNCIONES DE COMPLEJIDAD
# ============================================================================

class ComplexityFunction:
    """Representa una función de complejidad matemática"""
    
    FUNCTIONS = {
        # Constante
        "1": lambda n: np.ones_like(n),
        "c": lambda n: np.ones_like(n),
        
        # Logarítmica
        "log(n)": lambda n: np.log2(np.maximum(n, 1)),
        "log n": lambda n: np.log2(np.maximum(n, 1)),
        "logn": lambda n: np.log2(np.maximum(n, 1)),
        "1×log(n)": lambda n: np.log2(np.maximum(n, 1)),
        "1*log(n)": lambda n: np.log2(np.maximum(n, 1)),
        
        # Lineal
        "n": lambda n: n,
        
        # Linearítmica
        "n×log(n)": lambda n: n * np.log2(np.maximum(n, 1)),
        "nlog(n)": lambda n: n * np.log2(np.maximum(n, 1)),
        "n log n": lambda n: n * np.log2(np.maximum(n, 1)),
        "n*log(n)": lambda n: n * np.log2(np.maximum(n, 1)),
        
        # Cuadrática
        "n²": lambda n: n**2,
        "n^2": lambda n: n**2,
        "n*n": lambda n: n**2,
        
        # Cúbica
        "n³": lambda n: n**3,
        "n^3": lambda n: n**3,
        
        # Exponencial
        "2^n": lambda n: 2**n,
        "2ⁿ": lambda n: 2**n,
        "φⁿ": lambda n: ((1 + np.sqrt(5))/2)**n,
        
        # Factorial (aproximación con Stirling)
        "n!": lambda n: np.exp(n * np.log(np.maximum(n, 1)) - n),
    }
    
    @staticmethod
    def parse_complexity(complexity_str: str) -> Optional[callable]:
        """
        Parsea string de complejidad y retorna función callable.
        
        Args:
            complexity_str: String como "O(n²)", "Θ(n log n)", etc.
            
        Returns:
            Función lambda que toma array de n y retorna valores
        """
        # Limpiar notación Big-O/Theta/Omega
        clean = complexity_str.strip()
        for prefix in ["O(", "Ω(", "Θ(", "o(", "ω(", "θ("]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):-1]  # Remover prefix y paréntesis final
                break
        
        clean = clean.strip()
        
        # Buscar en diccionario de funciones
        if clean in ComplexityFunction.FUNCTIONS:
            return ComplexityFunction.FUNCTIONS[clean]
        
        # Intentar parsear potencias: n^k
        match = re.match(r'n\^(\d+)', clean)
        if match:
            power = int(match.group(1))
            return lambda n: n**power
        
        # Intentar parsear exponenciales: a^n
        match = re.match(r'(\d+)\^n', clean)
        if match:
            base = int(match.group(1))
            return lambda n: base**n
        
        # Fallback: función constante
        return lambda n: np.ones_like(n)
    
    @staticmethod
    def get_max_n_for_complexity(complexity_str: str) -> int:
        """
        Determina el rango de n apropiado para graficar.
        
        Args:
            complexity_str: String de complejidad
            
        Returns:
            Valor máximo de n recomendado
        """
        clean = complexity_str.lower()
        
        if "!" in clean or "factorial" in clean:
            return 8  # Factorial crece muy rápido
        elif "2^n" in clean or "^n" in clean or "ⁿ" in clean:
            return 15  # Exponencial
        elif "n^3" in clean or "n³" in clean:
            return 30
        elif "n^2" in clean or "n²" in clean:
            return 50
        elif "log" in clean:
            return 1000  # Logarítmica crece lento
        else:
            return 200  # Default
    
    @staticmethod
    def should_use_log_scale(complexities: List[str]) -> bool:
        """
        Determina si usar escala logarítmica en Y.
        
        Args:
            complexities: Lista de strings de complejidad
            
        Returns:
            True si debe usar escala log
        """
        for c in complexities:
            clean = c.lower()
            if "^n" in clean or "ⁿ" in clean or "!" in clean:
                return True
        return False


# ============================================================================
# GRAFICADOR PRINCIPAL
# ============================================================================

class ComplexityPlotter:
    """Genera gráficos de funciones de complejidad"""
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Inicializa el graficador.
        
        Args:
            theme: Tema de Plotly ("plotly", "plotly_white", "plotly_dark")
        """
        self.theme = theme
        self.colors = {
            "worst": "#ef4444",    # Rojo
            "best": "#10b981",     # Verde
            "average": "#f59e0b",  # Amarillo/Naranja
        }
    
    def plot_single_case(
        self,
        complexity: str,
        case_type: str = "worst",
        n_max: Optional[int] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Grafica una sola función de complejidad.
        
        Args:
            complexity: String de complejidad (ej: "O(n²)")
            case_type: "worst", "best" o "average"
            n_max: Valor máximo de n (auto si None)
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        # Parsear función
        func = ComplexityFunction.parse_complexity(complexity)
        
        # Determinar rango de n
        if n_max is None:
            n_max = ComplexityFunction.get_max_n_for_complexity(complexity)
        
        # Generar datos
        n_values = np.linspace(1, n_max, 500)
        y_values = func(n_values)
        
        # Crear figura
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=n_values,
            y=y_values,
            mode='lines',
            name=complexity,
            line=dict(color=self.colors.get(case_type, "#6b7280"), width=3),
            hovertemplate=f'<b>n = %{{x:.0f}}</b><br>T(n) = %{{y:.2e}}<extra></extra>'
        ))
        
        # Configurar layout
        if title is None:
            title = f"Función de Complejidad: {complexity}"
        
        use_log = ComplexityFunction.should_use_log_scale([complexity])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title="Tamaño de Entrada (n)",
            yaxis_title="Operaciones T(n)",
            template=self.theme,
            hovermode='x unified',
            yaxis_type='log' if use_log else 'linear',
            showlegend=True,
            height=500,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def plot_three_cases(
        self,
        worst: str,
        best: str,
        average: str,
        n_max: Optional[int] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Grafica peor, mejor y caso promedio en un solo gráfico.
        
        Args:
            worst: Complejidad del peor caso
            best: Complejidad del mejor caso
            average: Complejidad del caso promedio
            n_max: Valor máximo de n
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        # Determinar rango de n
        complexities = [worst, best, average]
        if n_max is None:
            n_max = max(
                ComplexityFunction.get_max_n_for_complexity(c) 
                for c in complexities
            )
        
        n_values = np.linspace(1, n_max, 500)
        
        # Crear figura
        fig = go.Figure()
        
        # Agregar cada caso
        cases = [
            ("Peor Caso", worst, "worst"),
            ("Mejor Caso", best, "best"),
            ("Caso Promedio", average, "average"),
        ]
        
        for label, complexity, case_type in cases:
            func = ComplexityFunction.parse_complexity(complexity)
            y_values = func(n_values)
            
            fig.add_trace(go.Scatter(
                x=n_values,
                y=y_values,
                mode='lines',
                name=f"{label}: {complexity}",
                line=dict(color=self.colors[case_type], width=3),
                hovertemplate=f'<b>{label}</b><br>n = %{{x:.0f}}<br>T(n) = %{{y:.2e}}<extra></extra>'
            ))
        
        # Configurar layout
        if title is None:
            title = "Comparación de Complejidades"
        
        use_log = ComplexityFunction.should_use_log_scale(complexities)
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            xaxis_title="Tamaño de Entrada (n)",
            yaxis_title="Número de Operaciones T(n)",
            template=self.theme,
            hovermode='x unified',
            yaxis_type='log' if use_log else 'linear',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.8)",
                font=dict(color="white")
            ),
            height=600,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def plot_comparison_multiple_algorithms(
        self,
        algorithms: Dict[str, str],
        n_max: Optional[int] = None,
        title: str = "Comparación de Algoritmos"
    ) -> go.Figure:
        """
        Compara múltiples algoritmos en un gráfico.
        
        Args:
            algorithms: Dict {nombre: complejidad}
            n_max: Valor máximo de n
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        if n_max is None:
            n_max = max(
                ComplexityFunction.get_max_n_for_complexity(c)
                for c in algorithms.values()
            )
        
        n_values = np.linspace(1, n_max, 500)
        
        fig = go.Figure()
        
        # Colores para múltiples algoritmos
        colors = [
            "#ef4444", "#10b981", "#f59e0b", "#3b82f6", 
            "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"
        ]
        
        for i, (name, complexity) in enumerate(algorithms.items()):
            func = ComplexityFunction.parse_complexity(complexity)
            y_values = func(n_values)
            
            fig.add_trace(go.Scatter(
                x=n_values,
                y=y_values,
                mode='lines',
                name=f"{name}: {complexity}",
                line=dict(color=colors[i % len(colors)], width=2.5),
                hovertemplate=f'<b>{name}</b><br>n = %{{x:.0f}}<br>T(n) = %{{y:.2e}}<extra></extra>'
            ))
        
        use_log = ComplexityFunction.should_use_log_scale(list(algorithms.values()))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            xaxis_title="Tamaño de Entrada (n)",
            yaxis_title="Operaciones T(n)",
            template=self.theme,
            hovermode='x unified',
            yaxis_type='log' if use_log else 'linear',
            showlegend=True,
            height=600,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return fig


# ============================================================================
# API SIMPLIFICADA
# ============================================================================

def plot_complexity(
    worst: str,
    best: str,
    average: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    API simple para graficar tres casos.
    
    Args:
        worst: Complejidad del peor caso
        best: Complejidad del mejor caso
        average: Complejidad del caso promedio
        title: Título del gráfico
        
    Returns:
        Figura de Plotly lista para mostrar
    
    Example:
        >>> fig = plot_complexity("O(n²)", "Ω(n)", "Θ(n²)")
        >>> fig.show()
    """
    plotter = ComplexityPlotter()
    return plotter.plot_three_cases(worst, best, average, title=title)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    # Demo: Graficar diferentes complejidades
    
    print("="*70)
    print("DEMO: GRAFICADOR DE COMPLEJIDADES")
    print("="*70)
    
    plotter = ComplexityPlotter()
    
    # Ejemplo 1: Bubble Sort
    print("\n📊 Graficando Bubble Sort...")
    fig1 = plotter.plot_three_cases(
        worst="O(n²)",
        best="Ω(n²)",
        average="Θ(n²)",
        title="Bubble Sort - Análisis de Complejidad"
    )
    fig1.write_html("bubble_sort_complexity.html")
    print("✓ Guardado en bubble_sort_complexity.html")
    
    # Ejemplo 2: Merge Sort
    print("\n📊 Graficando Merge Sort...")
    fig2 = plotter.plot_three_cases(
        worst="O(n×log(n))",
        best="Ω(n)",
        average="Θ(n²)",
        title="Merge Sort - Análisis de Complejidad"
    )
    fig2.write_html("merge_sort_complexity.html")
    print("✓ Guardado en merge_sort_complexity.html")
    
    # Ejemplo 3: Comparación múltiple
    print("\n📊 Comparando algoritmos...")
    fig3 = plotter.plot_comparison_multiple_algorithms(
        algorithms={
            "Bubble Sort": "O(n²)",
            "Merge Sort": "O(n log n)",
            "Quick Sort (avg)": "O(n log n)",
            "Quick Sort (worst)": "O(n²)",
            "Linear Search": "O(n)",
            "Binary Search": "O(log n)"
        },
        title="Comparación de Algoritmos de Ordenamiento y Búsqueda"
    )
    fig3.write_html("algorithms_comparison.html")
    print("✓ Guardado en algorithms_comparison.html")
    
    print("\n✅ Demo completado. Abre los archivos .html en tu navegador.")