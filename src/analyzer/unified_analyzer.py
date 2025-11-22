"""
Analizador Unificado de Complejidad Computacional
=================================================

Integra análisis iterativo (ciclos) y recursivo en un único sistema.
Maneja algoritmos híbridos como QuickSort, MergeSort con optimizaciones, etc.

Características:
- Detección automática de recursión
- Análisis de ciclos anidados
- Combinación de complejidades iterativas + recursivas
- Generación de ecuaciones de recurrencia completas
- Notaciones O(n), Ω(n), Θ(n)

Autor: Sistema de Análisis de Complejidad
Fecha: 2025
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from syntax_tree.nodes import *
from analyzer.complexity_analyzer import BasicComplexityAnalyzer, ComplexityResult
from analyzer.recursion_analyzer import RecursionAnalyzerVisitor, RecurrenceEquation
from analyzer.recurrence_solver import solve_recurrence, RecurrenceSolution


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class UnifiedComplexityResult:
    """
    Resultado completo del análisis unificado.
    
    Incluye tanto análisis iterativo como recursivo.
    """
    procedure_name: str
    
    # Análisis iterativo (ciclos)
    iterative_worst: str = "O(1)"
    iterative_best: str = "Ω(1)"
    iterative_average: str = "Θ(1)"
    
    # Análisis recursivo (si aplica)
    is_recursive: bool = False
    recurrence_equation: Optional[str] = None
    recurrence_solution: Optional[RecurrenceSolution] = None
    
    # Análisis combinado (final)
    final_worst: str = "O(1)"
    final_best: str = "Ω(1)"
    final_average: str = "Θ(1)"
    
    # Metadatos
    algorithm_type: str = "iterative"  # iterative, recursive, hybrid
    explanation: str = ""
    steps: List[str] = field(default_factory=list)
    
    # Desglose detallado
    loop_analysis: Dict[str, str] = field(default_factory=dict)
    recursion_tree: Optional[str] = None
    
    def __str__(self):
        result = f"""
{'='*70}
Procedimiento: {self.procedure_name}
{'='*70}

TIPO DE ALGORITMO: {self.algorithm_type.upper()}

COMPLEJIDAD FINAL:
  Peor Caso (O):      {self.final_worst}
  Mejor Caso (Ω):     {self.final_best}
  Caso Promedio (Θ): {self.final_average}
"""
        
        if self.is_recursive:
            result += f"""
ANÁLISIS RECURSIVO:
  Ecuación de Recurrencia: {self.recurrence_equation}
"""
            if self.recurrence_solution:
                result += f"  Solución: {self.recurrence_solution.big_theta}\n"
        
        if self.iterative_worst != "O(1)":
            result += f"""
ANÁLISIS ITERATIVO:
  Componente iterativo: {self.iterative_worst}
"""
        
        result += f"\nEXPLICACIÓN:\n{self.explanation}\n"
        
        if self.steps:
            result += "\nPASOS DEL ANÁLISIS:\n"
            for i, step in enumerate(self.steps, 1):
                result += f"  {i}. {step}\n"
        
        result += "="*70
        return result
    
    def to_dict(self) -> dict:
        """Serializa a diccionario para API/JSON"""
        return {
            "procedure_name": self.procedure_name,
            "algorithm_type": self.algorithm_type,
            "complexity": {
                "worst_case": self.final_worst,
                "best_case": self.final_best,
                "average_case": self.final_average
            },
            "recursive": {
                "is_recursive": self.is_recursive,
                "equation": self.recurrence_equation,
                "solution": self.recurrence_solution.to_dict() if self.recurrence_solution else None
            },
            "iterative": {
                "worst": self.iterative_worst,
                "best": self.iterative_best,
                "average": self.iterative_average
            },
            "explanation": self.explanation,
            "steps": self.steps
        }


# ============================================================================
# ANALIZADOR UNIFICADO
# ============================================================================

class UnifiedComplexityAnalyzer:
    """
    Analizador que combina análisis iterativo y recursivo.
    
    Flujo de análisis:
    1. Detectar si el algoritmo es recursivo
    2. Analizar componente iterativo (ciclos)
    3. Analizar componente recursivo (ecuaciones de recurrencia)
    4. Combinar ambos análisis
    5. Resolver ecuaciones de recurrencia
    6. Generar resultado unificado
    """
    
    def __init__(self):
        self.iterative_analyzer = BasicComplexityAnalyzer()
        self.results: Dict[str, UnifiedComplexityResult] = {}
    
    # ========================================================================
    # PUNTO DE ENTRADA PRINCIPAL
    # ========================================================================
    
    def analyze_program(self, program: ProgramNode) -> Dict[str, UnifiedComplexityResult]:
        """
        Analiza un programa completo.
        
        Args:
            program: AST del programa
            
        Returns:
            Dict con resultados por procedimiento
        """
        for procedure in program.procedures:
            result = self.analyze_procedure(procedure)
            self.results[procedure.name] = result
        
        return self.results
    
    def analyze_procedure(self, procedure: ProcedureNode) -> UnifiedComplexityResult:
        """
        Analiza un procedimiento individual.
        
        Estrategia:
        1. Análisis recursivo (si aplica)
        2. Análisis iterativo (ciclos)
        3. Combinación inteligente
        """
        steps = []
        steps.append(f"Analizando procedimiento: {procedure.name}")
        
        # ====================================================================
        # PASO 1: Detectar y analizar recursión
        # ====================================================================
        
        recursion_visitor = RecursionAnalyzerVisitor(procedure.name)
        recursion_result = recursion_visitor.visit_procedure(procedure)
        
        is_recursive = recursion_result.is_recursive
        recurrence_eq = None
        recurrence_solution = None
        
        if is_recursive:
            steps.append("✓ Algoritmo recursivo detectado")
            
            # Obtener ecuación de recurrencia
            if recursion_result.recurrence_equation:
                eq = recursion_result.recurrence_equation
                recurrence_eq = eq.worst_case_equation
                steps.append(f"  Ecuación: {recurrence_eq}")
                
                # Resolver ecuación con el solver
                try:
                    recurrence_solution = solve_recurrence(recurrence_eq)
                    steps.append(f"  Solución: {recurrence_solution.big_theta}")
                except Exception as e:
                    steps.append(f"  ⚠ No se pudo resolver automáticamente: {e}")
        else:
            steps.append("✓ Algoritmo iterativo (no recursivo)")
        
        # ====================================================================
        # PASO 2: Analizar componente iterativo
        # ====================================================================
        
        iterative_result = self.iterative_analyzer.analyze_procedure(procedure)
        
        steps.append(f"✓ Análisis iterativo: {iterative_result.worst_case}")
        
        # ====================================================================
        # PASO 3: Combinar análisis
        # ====================================================================
        
        if is_recursive and recurrence_solution:
            # Caso recursivo puro o híbrido
            algorithm_type = self._classify_algorithm(
                is_recursive, 
                iterative_result.worst_case,
                recurrence_solution.big_o
            )
            
            final_complexity = self._combine_complexities(
                iterative_result,
                recurrence_solution,
                algorithm_type
            )
            
            explanation = self._generate_explanation(
                algorithm_type,
                iterative_result,
                recurrence_eq,
                recurrence_solution
            )
            
        else:
            # Caso iterativo puro
            algorithm_type = "iterative"
            final_complexity = (
                iterative_result.worst_case,
                iterative_result.best_case,
                iterative_result.average_case
            )
            
            explanation = f"Algoritmo iterativo con complejidad {iterative_result.worst_case}"
        
        # ====================================================================
        # PASO 4: Construir resultado unificado
        # ====================================================================
        
        result = UnifiedComplexityResult(
            procedure_name=procedure.name,
            iterative_worst=iterative_result.worst_case,
            iterative_best=iterative_result.best_case,
            iterative_average=iterative_result.average_case,
            is_recursive=is_recursive,
            recurrence_equation=recurrence_eq,
            recurrence_solution=recurrence_solution,
            final_worst=final_complexity[0],
            final_best=final_complexity[1],
            final_average=final_complexity[2],
            algorithm_type=algorithm_type,
            explanation=explanation,
            steps=steps
        )
        
        return result
    
    # ========================================================================
    # MÉTODOS DE ANÁLISIS AUXILIARES
    # ========================================================================
    
    def _classify_algorithm(
        self, 
        is_recursive: bool, 
        iterative_complexity: str,
        recursive_complexity: str
    ) -> str:
        """
        Clasifica el algoritmo según sus características.
        
        Returns:
            "iterative", "recursive", "hybrid"
        """
        if not is_recursive:
            return "iterative"
        
        # Extraer orden de complejidad iterativa
        iterative_order = self._extract_order(iterative_complexity)
        
        # Si la parte iterativa es significativa (no constante)
        if iterative_order not in ["1", "O(1)"]:
            return "hybrid"
        else:
            return "recursive"
    
    def _combine_complexities(
        self,
        iterative: ComplexityResult,
        recursive: RecurrenceSolution,
        algorithm_type: str
    ) -> Tuple[str, str, str]:
        """
        Combina complejidades iterativa y recursiva.
        
        Estrategias:
        - Híbrido: Multiplicar o sumar según contexto
        - Recursivo puro: Usar solo la recursiva
        - Iterativo puro: Usar solo la iterativa
        
        Returns:
            (worst, best, average)
        """
        if algorithm_type == "iterative":
            return (
                iterative.worst_case,
                iterative.best_case,
                iterative.average_case
            )
        
        elif algorithm_type == "recursive":
            # Usar solo la solución recursiva
            return (
                recursive.big_o,
                recursive.big_omega,
                recursive.big_theta
            )
        
        else:  # hybrid
            # Combinar: max(iterativo, recursivo)
            # Ej: QuickSort con partition O(n) + recursión T(n) = 2T(n/2)
            # Resultado: O(n log n)
            
            worst = self._max_complexity(
                iterative.worst_case,
                recursive.big_o
            )
            
            best = self._max_complexity(
                iterative.best_case,
                recursive.big_omega
            )
            
            average = self._max_complexity(
                iterative.average_case,
                recursive.big_theta
            )
            
            return (worst, best, average)
    
    def _max_complexity(self, comp1: str, comp2: str) -> str:
        """Retorna la complejidad mayor entre dos"""
        order = {
            "O(1)": 0, "Ω(1)": 0, "Θ(1)": 0,
            "O(log(n))": 1, "Ω(log(n))": 1, "Θ(log(n))": 1,
            "O(n)": 2, "Ω(n)": 2, "Θ(n)": 2,
            "O(n×log(n))": 3, "Ω(n×log(n))": 3, "Θ(n×log(n))": 3,
            "O(n²)": 4, "Ω(n²)": 4, "Θ(n²)": 4,
            "O(n³)": 5, "Ω(n³)": 5, "Θ(n³)": 5,
            "O(2ⁿ)": 6, "Ω(2ⁿ)": 6, "Θ(2ⁿ)": 6,
        }
        
        # Normalizar notación
        comp1_norm = comp1.replace("O(", "O(").replace("Ω(", "Ω(").replace("Θ(", "Θ(")
        comp2_norm = comp2.replace("O(", "O(").replace("Ω(", "Ω(").replace("Θ(", "Θ(")
        
        val1 = order.get(comp1_norm, 2)
        val2 = order.get(comp2_norm, 2)
        
        return comp1 if val1 >= val2 else comp2
    
    def _extract_order(self, complexity: str) -> str:
        """Extrae el orden de una notación de complejidad"""
        for prefix in ["O(", "Ω(", "Θ("]:
            if complexity.startswith(prefix):
                return complexity[len(prefix):-1]
        return complexity
    
    def _generate_explanation(
        self,
        algorithm_type: str,
        iterative: ComplexityResult,
        recurrence_eq: Optional[str],
        recurrence_sol: Optional[RecurrenceSolution]
    ) -> str:
        """Genera explicación detallada del análisis"""
        
        if algorithm_type == "iterative":
            return f"Algoritmo puramente iterativo. {iterative.explanation}"
        
        elif algorithm_type == "recursive":
            explanation = f"Algoritmo recursivo.\n"
            explanation += f"Ecuación de recurrencia: {recurrence_eq}\n"
            
            if recurrence_sol:
                explanation += f"Solución: {recurrence_sol.big_theta}\n"
                explanation += f"Método usado: {recurrence_sol.method_used}\n"
                
                if recurrence_sol.steps:
                    explanation += "\nPasos de resolución:\n"
                    for step in recurrence_sol.steps[:3]:  # Primeros 3 pasos
                        explanation += f"  • {step}\n"
            
            return explanation
        
        else:  # hybrid
            explanation = f"Algoritmo híbrido (iterativo + recursivo).\n\n"
            explanation += f"Componente iterativo: {iterative.worst_case}\n"
            explanation += f"  {iterative.explanation}\n\n"
            explanation += f"Componente recursivo: {recurrence_eq}\n"
            
            if recurrence_sol:
                explanation += f"  Solución: {recurrence_sol.big_theta}\n"
            
            explanation += f"\nLa complejidad final es dominada por el componente de mayor orden."
            
            return explanation


# ============================================================================
# API PÚBLICA
# ============================================================================

def analyze_complexity_unified(ast: ProgramNode) -> Dict[str, UnifiedComplexityResult]:
    """
    API principal: Analiza complejidad de un programa completo.
    
    Args:
        ast: AST del programa parseado
        
    Returns:
        Dict con resultados por procedimiento
        
    Example:
        >>> from parser.parser import parse
        >>> ast = parse(code)
        >>> results = analyze_complexity_unified(ast)
        >>> print(results['QuickSort'])
    """
    analyzer = UnifiedComplexityAnalyzer()
    return analyzer.analyze_program(ast)


# ============================================================================
# DEMO Y TESTS
# ============================================================================

def demo():
    """Demuestra el analizador unificado con varios ejemplos"""
    
    from parser.parser import parse
    
    examples = {
        "Bubble Sort (Iterativo Puro)": """
BubbleSort(A[], n)
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
end
        """,
        
        "Merge Sort (Recursivo Puro)": """
MergeSort(A[], p, r)
begin
    if (p < r) then
    begin
        q ← floor((p + r) / 2)
        call MergeSort(A, p, q)
        call MergeSort(A, q+1, r)
        call Merge(A, p, q, r)
    end
end
        """,
        
        "QuickSort (Híbrido)": """
QuickSort(A[], p, r)
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
    
    A[r] ← A[i+1]
    A[i+1] ← pivot
    return i+1
end
        """,
        
        "Binary Search (Recursivo)": """
BinarySearch(A[], left, right, x)
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
end
        """
    }
    
    print("="*70)
    print("DEMOSTRACIÓN: ANALIZADOR UNIFICADO")
    print("="*70)
    
    for name, code in examples.items():
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print(f"{'='*70}")
        
        try:
            # Parsear código
            ast = parse(code)
            
            # Analizar con sistema unificado
            results = analyze_complexity_unified(ast)
            
            # Mostrar resultados
            for proc_name, result in results.items():
                print(result)
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()