"""
Analizador Unificado de Complejidad Computacional - VERSIÓN MEJORADA
====================================================================

MEJORA CRÍTICA: Proporciona ecuaciones de recurrencia completas para
mejor, peor y caso promedio, con sus soluciones detalladas.

Características:
- Detección automática de recursión
- Análisis de ciclos anidados
- Ecuaciones de recurrencia para todos los casos
- Resolución de ecuaciones con múltiples métodos
- Notaciones O(n), Ω(n), Θ(n) completas
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
# ESTRUCTURAS DE DATOS MEJORADAS
# ============================================================================

@dataclass
class RecurrenceAnalysis:
    """Análisis completo de ecuaciones de recurrencia"""
    
    # Ecuaciones para cada caso
    worst_case_equation: str = ""
    best_case_equation: str = ""
    average_case_equation: str = ""
    
    # Soluciones detalladas
    worst_case_solution: Optional[RecurrenceSolution] = None
    best_case_solution: Optional[RecurrenceSolution] = None
    average_case_solution: Optional[RecurrenceSolution] = None
    
    # Explicaciones
    worst_case_explanation: str = ""
    best_case_explanation: str = ""
    average_case_explanation: str = ""
    
    def __str__(self):
        result = "\n" + "="*70 + "\n"
        result += "ANÁLISIS DE RECURRENCIA COMPLETO\n"
        result += "="*70 + "\n"
        
        # PEOR CASO
        result += "\n🔴 PEOR CASO:\n"
        result += f"  Ecuación: {self.worst_case_equation}\n"
        if self.worst_case_solution:
            result += f"  Método: {self.worst_case_solution.method_used}\n"
            result += f"  Solución: {self.worst_case_solution.big_theta}\n"
            result += f"  Explicación: {self.worst_case_explanation}\n"
            if self.worst_case_solution.steps:
                result += f"\n  Pasos de resolución:\n"
                for i, step in enumerate(self.worst_case_solution.steps[:5], 1):
                    result += f"    {i}. {step}\n"
        
        # MEJOR CASO
        result += "\n🟢 MEJOR CASO:\n"
        result += f"  Ecuación: {self.best_case_equation}\n"
        if self.best_case_solution:
            result += f"  Método: {self.best_case_solution.method_used}\n"
            result += f"  Solución: {self.best_case_solution.big_theta}\n"
            result += f"  Explicación: {self.best_case_explanation}\n"
            if self.best_case_solution.steps:
                result += f"\n  Pasos de resolución:\n"
                for i, step in enumerate(self.best_case_solution.steps[:5], 1):
                    result += f"    {i}. {step}\n"
        
        # CASO PROMEDIO
        result += "\n🟡 CASO PROMEDIO:\n"
        result += f"  Ecuación: {self.average_case_equation}\n"
        if self.average_case_solution:
            result += f"  Método: {self.average_case_solution.method_used}\n"
            result += f"  Solución: {self.average_case_solution.big_theta}\n"
            result += f"  Explicación: {self.average_case_explanation}\n"
            if self.average_case_solution.steps:
                result += f"\n  Pasos de resolución:\n"
                for i, step in enumerate(self.average_case_solution.steps[:5], 1):
                    result += f"    {i}. {step}\n"
        
        result += "\n" + "="*70
        return result
    
    def to_dict(self) -> dict:
        """Serializa a diccionario"""
        return {
            "worst_case": {
                "equation": self.worst_case_equation,
                "solution": self.worst_case_solution.to_dict() if self.worst_case_solution else None,
                "explanation": self.worst_case_explanation
            },
            "best_case": {
                "equation": self.best_case_equation,
                "solution": self.best_case_solution.to_dict() if self.best_case_solution else None,
                "explanation": self.best_case_explanation
            },
            "average_case": {
                "equation": self.average_case_equation,
                "solution": self.average_case_solution.to_dict() if self.average_case_solution else None,
                "explanation": self.average_case_explanation
            }
        }


@dataclass
class UnifiedComplexityResult:
    """Resultado completo del análisis unificado - MEJORADO"""
    
    procedure_name: str
    
    # Análisis iterativo (ciclos)
    iterative_worst: str = "O(1)"
    iterative_best: str = "Ω(1)"
    iterative_average: str = "Θ(1)"
    
    # Análisis recursivo completo (NUEVO)
    is_recursive: bool = False
    recurrence_analysis: Optional[RecurrenceAnalysis] = None
    
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
        
        if self.is_recursive and self.recurrence_analysis:
            result += "\n" + str(self.recurrence_analysis)
        
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
                "recurrence_analysis": self.recurrence_analysis.to_dict() if self.recurrence_analysis else None
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
# ANALIZADOR UNIFICADO MEJORADO
# ============================================================================

class UnifiedComplexityAnalyzer:
    """Analizador con soporte completo para ecuaciones de recurrencia"""
    
    def __init__(self):
        self.iterative_analyzer = BasicComplexityAnalyzer()
        self.results: Dict[str, UnifiedComplexityResult] = {}
    
    def analyze_program(self, program: ProgramNode) -> Dict[str, UnifiedComplexityResult]:
        """Analiza un programa completo"""
        for procedure in program.procedures:
            result = self.analyze_procedure(procedure)
            self.results[procedure.name] = result
        
        return self.results
    
    def analyze_procedure(self, procedure: ProcedureNode) -> UnifiedComplexityResult:
        """Analiza un procedimiento individual - VERSIÓN MEJORADA"""
        
        steps = []
        steps.append(f"Analizando procedimiento: {procedure.name}")
        
        # ====================================================================
        # PASO 1: Detectar y analizar recursión
        # ====================================================================
        
        recursion_visitor = RecursionAnalyzerVisitor(procedure.name)
        recursion_result = recursion_visitor.visit_procedure(procedure)
        
        is_recursive = recursion_result.is_recursive
        recurrence_analysis = None
        
        if is_recursive:
            steps.append("✓ Algoritmo recursivo detectado")
            
            # NUEVO: Obtener ecuaciones para todos los casos
            if recursion_result.recurrence_equation:
                recurrence_eq_obj = recursion_result.recurrence_equation
                
                steps.append(f"  Generando ecuaciones para todos los casos...")
                
                recurrence_analysis = self._analyze_all_recurrence_cases(
                    recurrence_eq_obj,
                    steps
                )
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
        
        if is_recursive and recurrence_analysis:
            # Caso recursivo con ecuaciones completas
            algorithm_type = self._classify_algorithm(
                is_recursive,
                iterative_result.worst_case,
                recurrence_analysis.worst_case_solution.big_o if recurrence_analysis.worst_case_solution else "O(?)"
            )
            
            final_complexity = self._combine_complexities_enhanced(
                iterative_result,
                recurrence_analysis,
                algorithm_type
            )
            
            explanation = self._generate_explanation_enhanced(
                algorithm_type,
                iterative_result,
                recurrence_analysis
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
            recurrence_analysis=recurrence_analysis,
            final_worst=final_complexity[0],
            final_best=final_complexity[1],
            final_average=final_complexity[2],
            algorithm_type=algorithm_type,
            explanation=explanation,
            steps=steps
        )
        
        return result
    
    # ========================================================================
    # NUEVO: Análisis completo de recurrencia
    # ========================================================================
    
    def _analyze_all_recurrence_cases(
        self,
        recurrence_eq: RecurrenceEquation,
        steps: List[str]
    ) -> RecurrenceAnalysis:
        """
        Analiza y resuelve ecuaciones para todos los casos.
        
        NUEVO: Esta función es la clave para proporcionar análisis completo.
        """
        analysis = RecurrenceAnalysis()
        
        # PEOR CASO
        analysis.worst_case_equation = recurrence_eq.worst_case_equation
        analysis.worst_case_explanation = recurrence_eq.worst_case_explanation
        
        steps.append(f"  Peor caso: {analysis.worst_case_equation}")
        
        try:
            analysis.worst_case_solution = solve_recurrence(analysis.worst_case_equation)
            steps.append(f"    → Solución: {analysis.worst_case_solution.big_theta}")
            steps.append(f"    → Método: {analysis.worst_case_solution.method_used}")
        except Exception as e:
            steps.append(f"    ⚠ No se pudo resolver automáticamente: {e}")
        
        # MEJOR CASO
        analysis.best_case_equation = recurrence_eq.best_case_equation
        analysis.best_case_explanation = recurrence_eq.best_case_explanation
        
        steps.append(f"  Mejor caso: {analysis.best_case_equation}")
        
        try:
            analysis.best_case_solution = solve_recurrence(analysis.best_case_equation)
            steps.append(f"    → Solución: {analysis.best_case_solution.big_theta}")
            steps.append(f"    → Método: {analysis.best_case_solution.method_used}")
        except Exception as e:
            steps.append(f"    ⚠ No se pudo resolver automáticamente: {e}")
        
        # CASO PROMEDIO
        analysis.average_case_equation = recurrence_eq.average_case_equation
        analysis.average_case_explanation = recurrence_eq.average_case_explanation
        
        steps.append(f"  Caso promedio: {analysis.average_case_equation}")
        
        try:
            analysis.average_case_solution = solve_recurrence(analysis.average_case_equation)
            steps.append(f"    → Solución: {analysis.average_case_solution.big_theta}")
            steps.append(f"    → Método: {analysis.average_case_solution.method_used}")
        except Exception as e:
            steps.append(f"    ⚠ No se pudo resolver automáticamente: {e}")
        
        return analysis
    
    # ========================================================================
    # MÉTODOS AUXILIARES MEJORADOS
    # ========================================================================
    
    def _classify_algorithm(
        self,
        is_recursive: bool,
        iterative_complexity: str,
        recursive_complexity: str
    ) -> str:
        """Clasifica el algoritmo"""
        if not is_recursive:
            return "iterative"
        
        iterative_order = self._extract_order(iterative_complexity)
        
        if iterative_order not in ["1", "O(1)"]:
            return "hybrid"
        else:
            return "recursive"
    
    def _combine_complexities_enhanced(
        self,
        iterative: ComplexityResult,
        recurrence: RecurrenceAnalysis,
        algorithm_type: str
    ) -> Tuple[str, str, str]:
        """Combina complejidades usando soluciones de recurrencia"""
        
        if algorithm_type == "iterative":
            return (
                iterative.worst_case,
                iterative.best_case,
                iterative.average_case
            )
        
        elif algorithm_type == "recursive":
            # Usar soluciones de recurrencia
            worst = recurrence.worst_case_solution.big_o if recurrence.worst_case_solution else "O(?)"
            best = recurrence.best_case_solution.big_omega if recurrence.best_case_solution else "Ω(?)"
            average = recurrence.average_case_solution.big_theta if recurrence.average_case_solution else "Θ(?)"
            
            return (worst, best, average)
        
        else:  # hybrid
            # Combinar: max(iterativo, recursivo)
            worst_rec = recurrence.worst_case_solution.big_o if recurrence.worst_case_solution else "O(?)"
            best_rec = recurrence.best_case_solution.big_omega if recurrence.best_case_solution else "Ω(?)"
            avg_rec = recurrence.average_case_solution.big_theta if recurrence.average_case_solution else "Θ(?)"
            
            worst = self._max_complexity(iterative.worst_case, worst_rec)
            best = self._max_complexity(iterative.best_case, best_rec)
            average = self._max_complexity(iterative.average_case, avg_rec)
            
            return (worst, best, average)
    
    def _max_complexity(self, comp1: str, comp2: str) -> str:
        """Retorna la complejidad mayor"""
        order = {
            "O(1)": 0, "Ω(1)": 0, "Θ(1)": 0,
            "O(log(n))": 1, "Ω(log(n))": 1, "Θ(log(n))": 1,
            "O(n)": 2, "Ω(n)": 2, "Θ(n)": 2,
            "O(n×log(n))": 3, "Ω(n×log(n))": 3, "Θ(n×log(n))": 3,
            "O(n²)": 4, "Ω(n²)": 4, "Θ(n²)": 4,
            "O(n³)": 5, "Ω(n³)": 5, "Θ(n³)": 5,
            "O(2ⁿ)": 6, "Ω(2ⁿ)": 6, "Θ(2ⁿ)": 6,
        }
        
        val1 = order.get(comp1, 2)
        val2 = order.get(comp2, 2)
        
        return comp1 if val1 >= val2 else comp2
    
    def _extract_order(self, complexity: str) -> str:
        """Extrae el orden"""
        for prefix in ["O(", "Ω(", "Θ("]:
            if complexity.startswith(prefix):
                return complexity[len(prefix):-1]
        return complexity
    
    def _generate_explanation_enhanced(
        self,
        algorithm_type: str,
        iterative: ComplexityResult,
        recurrence: RecurrenceAnalysis
    ) -> str:
        """
        Genera explicación detallada - CORREGIDO
        
        NUEVO: Incluye TODOS los casos (worst, best, average) con pasos de resolución
        """
        
        if algorithm_type == "iterative":
            return f"Algoritmo puramente iterativo. {iterative.explanation}"
        
        elif algorithm_type == "recursive":
            explanation = "**Algoritmo recursivo**\n\n"
            
            # ============================================================
            # PEOR CASO
            # ============================================================
            explanation += "### 🔴 PEOR CASO\n\n"
            explanation += f"**Ecuación de recurrencia:**  \n`{recurrence.worst_case_equation}`\n\n"
            
            if recurrence.worst_case_solution:
                explanation += f"**Solución:**  \n{recurrence.worst_case_solution.big_theta}\n\n"
                explanation += f"**Método usado:**  \n{recurrence.worst_case_solution.method_used}\n\n"
                explanation += f"**Explicación:**  \n{recurrence.worst_case_explanation}\n\n"
                
                # NUEVO: Agregar pasos de resolución
                if recurrence.worst_case_solution.steps:
                    explanation += "**Pasos de resolución:**\n"
                    for i, step in enumerate(recurrence.worst_case_solution.steps, 1):
                        explanation += f"{i}. {step}\n"
                    explanation += "\n"
            
            # ============================================================
            # MEJOR CASO
            # ============================================================
            explanation += "### 🟢 MEJOR CASO\n\n"
            explanation += f"**Ecuación de recurrencia:**  \n`{recurrence.best_case_equation}`\n\n"
            
            if recurrence.best_case_solution:
                explanation += f"**Solución:**  \n{recurrence.best_case_solution.big_theta}\n\n"
                explanation += f"**Método usado:**  \n{recurrence.best_case_solution.method_used}\n\n"
                explanation += f"**Explicación:**  \n{recurrence.best_case_explanation}\n\n"
                
                # NUEVO: Agregar pasos de resolución
                if recurrence.best_case_solution.steps:
                    explanation += "**Pasos de resolución:**\n"
                    for i, step in enumerate(recurrence.best_case_solution.steps, 1):
                        explanation += f"{i}. {step}\n"
                    explanation += "\n"
            
            # ============================================================
            # CASO PROMEDIO - NUEVO
            # ============================================================
            explanation += "### 🟡 CASO PROMEDIO\n\n"
            explanation += f"**Ecuación de recurrencia:**  \n`{recurrence.average_case_equation}`\n\n"
            
            if recurrence.average_case_solution:
                explanation += f"**Solución:**  \n{recurrence.average_case_solution.big_theta}\n\n"
                explanation += f"**Método usado:**  \n{recurrence.average_case_solution.method_used}\n\n"
                explanation += f"**Explicación:**  \n{recurrence.average_case_explanation}\n\n"
                
                # NUEVO: Agregar pasos de resolución
                if recurrence.average_case_solution.steps:
                    explanation += "**Pasos de resolución:**\n"
                    for i, step in enumerate(recurrence.average_case_solution.steps, 1):
                        explanation += f"{i}. {step}\n"
                    explanation += "\n"
            
            return explanation
        
        else:  # hybrid
            explanation = "**Algoritmo híbrido** (iterativo + recursivo)\n\n"
            
            explanation += "**Componente iterativo:**  \n"
            explanation += f"{iterative.worst_case}  \n"
            explanation += f"{iterative.explanation}\n\n"
            
            explanation += "**Componente recursivo:**\n\n"
            
            # PEOR CASO
            explanation += "### 🔴 PEOR CASO\n\n"
            if recurrence.worst_case_solution:
                explanation += f"**Ecuación:** `{recurrence.worst_case_equation}`\n\n"
                explanation += f"**Solución:** {recurrence.worst_case_solution.big_theta}\n\n"
                
                if recurrence.worst_case_solution.steps:
                    explanation += "**Pasos:**\n"
                    for i, step in enumerate(recurrence.worst_case_solution.steps[:5], 1):
                        explanation += f"{i}. {step}\n"
                    explanation += "\n"
            
            # MEJOR CASO
            explanation += "### 🟢 MEJOR CASO\n\n"
            if recurrence.best_case_solution:
                explanation += f"**Ecuación:** `{recurrence.best_case_equation}`\n\n"
                explanation += f"**Solución:** {recurrence.best_case_solution.big_theta}\n\n"
                
                if recurrence.best_case_solution.steps:
                    explanation += "**Pasos:**\n"
                    for i, step in enumerate(recurrence.best_case_solution.steps[:5], 1):
                        explanation += f"{i}. {step}\n"
                    explanation += "\n"
            
            # CASO PROMEDIO - NUEVO
            explanation += "### 🟡 CASO PROMEDIO\n\n"
            if recurrence.average_case_solution:
                explanation += f"**Ecuación:** `{recurrence.average_case_equation}`\n\n"
                explanation += f"**Solución:** {recurrence.average_case_solution.big_theta}\n\n"
                
                if recurrence.average_case_solution.steps:
                    explanation += "**Pasos:**\n"
                    for i, step in enumerate(recurrence.average_case_solution.steps[:5], 1):
                        explanation += f"{i}. {step}\n"
                    explanation += "\n"
            
            return explanation


# ============================================================================
# API PÚBLICA
# ============================================================================

def analyze_complexity_unified(ast: ProgramNode) -> Dict[str, UnifiedComplexityResult]:
    """
    API principal: Analiza complejidad de un programa completo.
    
    MEJORA: Ahora incluye ecuaciones de recurrencia completas para todos los casos.
    """
    analyzer = UnifiedComplexityAnalyzer()
    return analyzer.analyze_program(ast)


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demuestra el analizador mejorado"""
    
    from parser.parser import parse
    
    examples = {
        "QuickSort (Híbrido - Mejor Demo)": """
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
    
    return i+1
end
        """,
        
        "Fibonacci (Recursivo Binario)": """
Fibonacci(n)
begin
    if (n ≤ 1) then
    begin
        return n
    end
    else
    begin
        return call Fibonacci(n-1) + call Fibonacci(n-2)
    end
end
        """
    }
    
    print("="*70)
    print("DEMOSTRACIÓN: ANALIZADOR UNIFICADO MEJORADO")
    print("="*70)
    
    for name, code in examples.items():
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print(f"{'='*70}")
        
        try:
            ast = parse(code)
            results = analyze_complexity_unified(ast)
            
            for proc_name, result in results.items():
                print(result)
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()