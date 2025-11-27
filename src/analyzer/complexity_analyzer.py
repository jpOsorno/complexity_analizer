"""
Analizador de Complejidad para Algoritmos Iterativos - VERSIÓN MEJORADA
=======================================================================

MEJORA CRÍTICA: Genera análisis detallado con sumatorias y explicaciones
paso a paso para mejor, peor y caso promedio.

Características:
- Detección de ciclos anidados con análisis de profundidad
- Generación de sumatorias matemáticas
- Explicación paso a paso del análisis
- Detección de early exit y condiciones
- Análisis de mejor, peor y caso promedio
"""

import sys
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from syntax_tree.nodes import *


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class LoopAnalysis:
    """Análisis detallado de un ciclo individual"""
    variable: str
    start: str
    end: str
    iterations: str  # Expresión simbólica: "n", "n-1", "n-i", etc.
    body_cost: str   # Costo del cuerpo: "O(1)", "O(n)", etc.
    has_early_exit: bool = False
    condition_for_exit: Optional[str] = None
    depth: int = 0   # Profundidad de anidamiento
    
    def to_summation(self) -> str:
        """Genera la sumatoria matemática del ciclo"""
        if self.has_early_exit:
            return f"Σ(i={self.start} to {self.end}) {self.body_cost} (con early exit posible)"
        else:
            return f"Σ(i={self.start} to {self.end}) {self.body_cost}"
    
    def __str__(self):
        exit_note = " [early exit]" if self.has_early_exit else ""
        return f"FOR {self.variable}={self.start} to {self.end}: {self.iterations} iteraciones{exit_note}"


@dataclass
class IterativeComplexityAnalysis:
    """Análisis completo de complejidad iterativa"""
    
    # Complejidades finales
    worst_case: str = "O(1)"
    best_case: str = "Ω(1)"
    average_case: str = "Θ(1)"
    
    # Análisis detallado por caso
    worst_case_explanation: str = ""
    best_case_explanation: str = ""
    average_case_explanation: str = ""
    
    # Sumatorias y ecuaciones
    worst_case_summation: str = ""
    best_case_summation: str = ""
    average_case_summation: str = ""
    
    # Pasos del análisis
    worst_case_steps: List[str] = field(default_factory=list)
    best_case_steps: List[str] = field(default_factory=list)
    average_case_steps: List[str] = field(default_factory=list)
    
    # Información estructural
    loops: List[LoopAnalysis] = field(default_factory=list)
    max_nesting_depth: int = 0
    has_conditionals: bool = False
    has_early_exit: bool = False
    
    def __str__(self):
        result = "\n" + "="*70 + "\n"
        result += "ANÁLISIS ITERATIVO COMPLETO\n"
        result += "="*70 + "\n"
        
        # PEOR CASO
        result += "\n🔴 PEOR CASO:\n"
        result += f"  Complejidad: {self.worst_case}\n"
        if self.worst_case_summation:
            result += f"  Sumatoria: {self.worst_case_summation}\n"
        result += f"  Explicación: {self.worst_case_explanation}\n"
        if self.worst_case_steps:
            result += "\n  Pasos:\n"
            for i, step in enumerate(self.worst_case_steps, 1):
                result += f"    {i}. {step}\n"
        
        # MEJOR CASO
        result += "\n🟢 MEJOR CASO:\n"
        result += f"  Complejidad: {self.best_case}\n"
        if self.best_case_summation:
            result += f"  Sumatoria: {self.best_case_summation}\n"
        result += f"  Explicación: {self.best_case_explanation}\n"
        if self.best_case_steps:
            result += "\n  Pasos:\n"
            for i, step in enumerate(self.best_case_steps, 1):
                result += f"    {i}. {step}\n"
        
        # CASO PROMEDIO
        result += "\n🟡 CASO PROMEDIO:\n"
        result += f"  Complejidad: {self.average_case}\n"
        if self.average_case_summation:
            result += f"  Sumatoria: {self.average_case_summation}\n"
        result += f"  Explicación: {self.average_case_explanation}\n"
        if self.average_case_steps:
            result += "\n  Pasos:\n"
            for i, step in enumerate(self.average_case_steps, 1):
                result += f"    {i}. {step}\n"
        
        result += "\n" + "="*70
        return result
    
    def to_dict(self) -> dict:
        """Serializa a diccionario"""
        return {
            "worst_case": {
                "complexity": self.worst_case,
                "summation": self.worst_case_summation,
                "explanation": self.worst_case_explanation,
                "steps": self.worst_case_steps
            },
            "best_case": {
                "complexity": self.best_case,
                "summation": self.best_case_summation,
                "explanation": self.best_case_explanation,
                "steps": self.best_case_steps
            },
            "average_case": {
                "complexity": self.average_case,
                "summation": self.average_case_summation,
                "explanation": self.average_case_explanation,
                "steps": self.average_case_steps
            },
            "structure": {
                "max_nesting_depth": self.max_nesting_depth,
                "num_loops": len(self.loops),
                "has_conditionals": self.has_conditionals,
                "has_early_exit": self.has_early_exit
            }
        }


# ============================================================================
# ANALIZADOR MEJORADO
# ============================================================================

class EnhancedComplexityAnalyzer:
    """
    Analizador de complejidad iterativa con análisis detallado.
    
    MEJORA: Genera sumatorias, explicaciones y pasos para cada caso.
    """
    
    def __init__(self):
        self.current_procedure = None
        self.current_depth = 0
        self.loops_stack: List[LoopAnalysis] = []
        self.all_loops: List[LoopAnalysis] = []
        self.has_early_exit = False
        self.has_conditionals = False
    
    # ========================================================================
    # ANÁLISIS PRINCIPAL
    # ========================================================================
    
    def analyze_procedure(self, procedure: ProcedureNode) -> IterativeComplexityAnalysis:
        """Analiza un procedimiento completo"""
        self.current_procedure = procedure.name
        self.current_depth = 0
        self.loops_stack = []
        self.all_loops = []
        self.has_early_exit = False
        self.has_conditionals = False
        
        # Analizar el cuerpo
        self._analyze_block(procedure.body)
        
        # Generar análisis para los 3 casos
        return self._build_complete_analysis()
    
    def _analyze_block(self, block: BlockNode):
        """Analiza un bloque de código"""
        for stmt in block.statements:
            self._analyze_statement(stmt)
    
    def _analyze_statement(self, stmt: StatementNode):
        """Analiza una sentencia"""
        if isinstance(stmt, ForNode):
            self._analyze_for(stmt)
        elif isinstance(stmt, WhileNode):
            self._analyze_while(stmt)
        elif isinstance(stmt, RepeatNode):
            self._analyze_repeat(stmt)
        elif isinstance(stmt, IfNode):
            self._analyze_if(stmt)
        elif isinstance(stmt, ReturnNode):
            if self.current_depth > 0:
                self.has_early_exit = True
    
    # ========================================================================
    # ANÁLISIS DE CICLOS
    # ========================================================================
    
    def _analyze_for(self, node: ForNode):
        """Analiza un ciclo FOR"""
        # Extraer información del ciclo
        variable = node.variable
        start = self._expr_to_string(node.start)
        end = self._expr_to_string(node.end)
        
        # Calcular iteraciones
        iterations = self._calculate_iterations(start, end, variable)
        
        # Detectar early exit en el cuerpo
        has_exit = self._has_early_exit(node.body)
        
        # Crear análisis del ciclo
        loop = LoopAnalysis(
            variable=variable,
            start=start,
            end=end,
            iterations=iterations,
            body_cost="O(1)",  # Se actualizará después
            has_early_exit=has_exit,
            depth=self.current_depth
        )
        
        # Agregar al stack y lista
        self.loops_stack.append(loop)
        self.all_loops.append(loop)
        
        # Analizar cuerpo (incrementar profundidad)
        self.current_depth += 1
        self._analyze_block(node.body)
        self.current_depth -= 1
        
        # Actualizar costo del cuerpo basado en ciclos internos
        if self.current_depth == 0:
            # Es el ciclo más externo, calcular costo acumulado
            loop.body_cost = self._calculate_body_cost(node.body)
        
        self.loops_stack.pop()
    
    def _analyze_while(self, node: WhileNode):
        """Analiza WHILE (similar a FOR pero con iteraciones indeterminadas)"""
        has_exit = self._has_early_exit(node.body)
        
        loop = LoopAnalysis(
            variable="while",
            start="0",
            end="n",
            iterations="n (indeterminado)",
            body_cost="O(1)",
            has_early_exit=has_exit,
            depth=self.current_depth
        )
        
        self.loops_stack.append(loop)
        self.all_loops.append(loop)
        
        self.current_depth += 1
        self._analyze_block(node.body)
        self.current_depth -= 1
        
        self.loops_stack.pop()
    
    def _analyze_repeat(self, node: RepeatNode):
        """Analiza REPEAT-UNTIL"""
        has_exit = True  # REPEAT siempre ejecuta al menos una vez
        
        loop = LoopAnalysis(
            variable="repeat",
            start="1",
            end="n",
            iterations="n (indeterminado)",
            body_cost="O(1)",
            has_early_exit=has_exit,
            depth=self.current_depth
        )
        
        self.loops_stack.append(loop)
        self.all_loops.append(loop)
        
        self.current_depth += 1
        self._analyze_block(node.body)
        self.current_depth -= 1
        
        self.loops_stack.pop()
    
    def _analyze_if(self, node: IfNode):
        """Analiza condicionales"""
        self.has_conditionals = True
        
        # Analizar ambas ramas
        self._analyze_block(node.then_block)
        if node.else_block:
            self._analyze_block(node.else_block)
    
    # ========================================================================
    # CÁLCULOS DE ITERACIONES Y COSTOS
    # ========================================================================
    
    def _calculate_iterations(self, start: str, end: str, variable: str) -> str:
        """
        Calcula el número de iteraciones de un ciclo.
        
        Ejemplos:
        - start=1, end=n → "n"
        - start=1, end=n-1 → "n-1"
        - start=i, end=n → "n-i+1"
        - start=1, end=n-i → "n-i"
        """
        # Normalizar
        start_clean = start.strip()
        end_clean = end.strip()
        
        # Caso simple: start=1 o 0
        if start_clean in ["0", "1"]:
            if end_clean == "n":
                return "n"
            elif "n-1" in end_clean:
                return "n-1" if start_clean == "1" else "n"
            elif "n-i" in end_clean:
                return "n-i" if start_clean == "1" else "n-i+1"
            elif end_clean.replace(" ", "") == "n-1":
                return "n-1" if start_clean == "1" else "n"
            else:
                return end_clean
        
        # Caso: start=variable (ej: i)
        if start_clean.isalpha() and len(start_clean) == 1:
            if end_clean == "n":
                return f"n-{start_clean}+1"
            elif "n-" in end_clean:
                return f"{end_clean}-{start_clean}+1"
            else:
                return f"{end_clean}-{start_clean}+1"
        
        # Fallback
        return f"{end_clean}-{start_clean}+1"
    
    def _calculate_body_cost(self, block: BlockNode) -> str:
        """Calcula el costo del cuerpo de un ciclo"""
        # Contar ciclos internos
        inner_loops = sum(1 for stmt in block.statements if isinstance(stmt, (ForNode, WhileNode, RepeatNode)))
        
        if inner_loops == 0:
            return "O(1)"
        elif inner_loops == 1:
            return "O(n)"
        elif inner_loops == 2:
            return "O(n²)"
        else:
            return f"O(n^{inner_loops})"
    
    def _has_early_exit(self, block: BlockNode) -> bool:
        """Detecta si hay early exit en un bloque"""
        for stmt in block.statements:
            if isinstance(stmt, ReturnNode):
                return True
            if isinstance(stmt, IfNode):
                if self._block_has_return(stmt.then_block):
                    return True
                if stmt.else_block and self._block_has_return(stmt.else_block):
                    return True
        return False
    
    def _block_has_return(self, block: BlockNode) -> bool:
        """Verifica si un bloque contiene return"""
        for stmt in block.statements:
            if isinstance(stmt, ReturnNode):
                return True
            if isinstance(stmt, IfNode):
                if self._block_has_return(stmt.then_block):
                    return True
                if stmt.else_block and self._block_has_return(stmt.else_block):
                    return True
        return False
    
    def _expr_to_string(self, expr: ExpressionNode) -> str:
        """Convierte expresión a string"""
        if isinstance(expr, NumberNode):
            return str(expr.value)
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._expr_to_string(expr.left)
            right = self._expr_to_string(expr.right)
            return f"{left}{expr.op}{right}"
        else:
            return "?"
    
    # ========================================================================
    # CONSTRUCCIÓN DEL ANÁLISIS COMPLETO
    # ========================================================================
    
    def _build_complete_analysis(self) -> IterativeComplexityAnalysis:
        """Construye el análisis completo para los 3 casos"""
        
        analysis = IterativeComplexityAnalysis()
        analysis.loops = self.all_loops
        analysis.max_nesting_depth = max((loop.depth for loop in self.all_loops), default=0) + 1
        analysis.has_conditionals = self.has_conditionals
        analysis.has_early_exit = self.has_early_exit
        
        # ====================================================================
        # PEOR CASO
        # ====================================================================
        
        worst_steps = []
        worst_steps.append("Análisis del peor caso (todos los ciclos ejecutan máximo de iteraciones)")
        
        # Generar sumatoria
        if len(self.all_loops) == 0:
            analysis.worst_case = "O(1)"
            analysis.worst_case_summation = "No hay ciclos"
            analysis.worst_case_explanation = "No hay ciclos, solo operaciones constantes"
            worst_steps.append("Sin ciclos → O(1)")
        
        elif len(self.all_loops) == 1:
            loop = self.all_loops[0]
            analysis.worst_case_summation = loop.to_summation()
            analysis.worst_case = f"O({loop.iterations})"
            analysis.worst_case_explanation = f"Un solo ciclo de {loop.iterations} iteraciones"
            worst_steps.append(f"Ciclo: for {loop.variable} = {loop.start} to {loop.end}")
            worst_steps.append(f"Iteraciones: {loop.iterations}")
            worst_steps.append(f"Costo por iteración: O(1)")
            worst_steps.append(f"Total: {loop.iterations} × O(1) = O({loop.iterations})")
        
        else:
            # Ciclos anidados
            analysis.worst_case_summation = self._generate_nested_summation(self.all_loops)
            complexity = self._calculate_nested_complexity(self.all_loops)
            analysis.worst_case = f"O({complexity})"
            
            worst_steps.append(f"Ciclos anidados detectados (profundidad: {analysis.max_nesting_depth})")
            
            for i, loop in enumerate(self.all_loops, 1):
                indent = "  " * loop.depth
                worst_steps.append(f"{indent}Ciclo {i}: for {loop.variable} = {loop.start} to {loop.end}")
                worst_steps.append(f"{indent}  → {loop.iterations} iteraciones")
            
            worst_steps.append(f"\nMultiplicar iteraciones: {self._format_multiplication(self.all_loops)}")
            worst_steps.append(f"Simplificar: O({complexity})")
            
            analysis.worst_case_explanation = self._explain_nested_loops(self.all_loops, "peor")
        
        analysis.worst_case_steps = worst_steps
        
        # ====================================================================
        # MEJOR CASO
        # ====================================================================
        
        best_steps = []
        
        if self.has_early_exit:
            best_steps.append("Análisis del mejor caso (early exit detectado)")
            analysis.best_case = "Ω(1)"
            analysis.best_case_summation = "Early exit en primera iteración"
            analysis.best_case_explanation = "Con early exit, el algoritmo puede terminar en la primera iteración"
            best_steps.append("Early exit posible → Ω(1)")
        else:
            best_steps.append("Análisis del mejor caso (sin early exit, igual al peor caso)")
            analysis.best_case = analysis.worst_case.replace("O", "Ω")
            analysis.best_case_summation = analysis.worst_case_summation
            analysis.best_case_explanation = "Sin early exit, el mejor caso es igual al peor caso"
            best_steps.extend([f"  {step}" for step in worst_steps[1:]])
        
        analysis.best_case_steps = best_steps
        
        # ====================================================================
        # CASO PROMEDIO
        # ====================================================================
        
        avg_steps = []
        avg_steps.append("Análisis del caso promedio")
        
        if self.has_early_exit:
            # Con early exit, el promedio está entre O(1) y O(worst)
            analysis.average_case = analysis.worst_case.replace("O", "Θ")
            analysis.average_case_summation = "Promedio entre mejor y peor caso"
            analysis.average_case_explanation = f"En promedio, con early exit, la complejidad es {analysis.average_case}"
            avg_steps.append(f"Con early exit: Θ(n/2) ≈ {analysis.average_case}")
        else:
            # Sin early exit, promedio = peor caso
            analysis.average_case = analysis.worst_case.replace("O", "Θ")
            analysis.average_case_summation = analysis.worst_case_summation
            analysis.average_case_explanation = "Sin early exit, el caso promedio es igual al peor caso"
            avg_steps.append(f"Sin early exit → Caso promedio = Peor caso")
            avg_steps.append(f"Resultado: {analysis.average_case}")
        
        analysis.average_case_steps = avg_steps
        
        return analysis
    
    # ========================================================================
    # GENERADORES DE SUMATORIAS Y EXPLICACIONES
    # ========================================================================
    
    def _generate_nested_summation(self, loops: List[LoopAnalysis]) -> str:
        """Genera la sumatoria anidada"""
        if len(loops) == 0:
            return "1"
        elif len(loops) == 1:
            return loops[0].to_summation()
        else:
            # Construir sumatorias anidadas
            result = ""
            for loop in sorted(loops, key=lambda l: l.depth):
                result += f"Σ({loop.variable}={loop.start} to {loop.end}) "
            result += "1"
            return result
    
    def _calculate_nested_complexity(self, loops: List[LoopAnalysis]) -> str:
        """Calcula la complejidad de ciclos anidados"""
        if len(loops) == 0:
            return "1"
        
        # Contar potencia de n
        n_power = 0
        has_dependent = False
        
        for loop in loops:
            if "n" in loop.iterations:
                if any(var in loop.iterations for var in [l.variable for l in loops if l.depth < loop.depth]):
                    # Depende de variable externa
                    has_dependent = True
                n_power += 1
        
        if n_power == 0:
            return "1"
        elif n_power == 1:
            return "n"
        elif n_power == 2:
            return "n²"
        elif n_power == 3:
            return "n³"
        else:
            return f"n^{n_power}"
    
    def _format_multiplication(self, loops: List[LoopAnalysis]) -> str:
        """Formatea la multiplicación de iteraciones"""
        iterations = [loop.iterations for loop in loops]
        return " × ".join(iterations)
    
    def _explain_nested_loops(self, loops: List[LoopAnalysis], case_type: str) -> str:
        """Genera explicación para ciclos anidados"""
        if len(loops) <= 1:
            return "Ciclo simple"
        
        explanation = f"Ciclos anidados (profundidad {max(l.depth for l in loops) + 1}):\n"
        
        for i, loop in enumerate(loops, 1):
            indent = "  " * loop.depth
            explanation += f"{indent}• Ciclo {i}: {loop.iterations} iteraciones\n"
        
        complexity = self._calculate_nested_complexity(loops)
        explanation += f"\nComplejidad final: O({complexity})"
        
        return explanation


# ============================================================================
# API PÚBLICA (COMPATIBILIDAD CON unified_analyzer)
# ============================================================================

class BasicComplexityAnalyzer:
    """Wrapper para mantener compatibilidad con código existente"""
    
    def __init__(self):
        self.enhanced = EnhancedComplexityAnalyzer()
    
    def analyze_procedure(self, procedure: ProcedureNode) -> IterativeComplexityAnalysis:
        """Analiza un procedimiento (retorna análisis detallado)"""
        return self.enhanced.analyze_procedure(procedure)


# ============================================================================
# COMPATIBILIDAD: ComplexityResult (DEPRECATED)
# ============================================================================

@dataclass
class ComplexityResult:
    """
    DEPRECATED: Mantener para compatibilidad.
    Usar IterativeComplexityAnalysis en su lugar.
    """
    worst_case: str = "O(1)"
    best_case: str = "Ω(1)"
    average_case: str = "Θ(1)"
    exact_cost: Optional[str] = None
    explanation: str = ""
    steps: List[str] = field(default_factory=list)


def analyze_complexity(ast: ProgramNode) -> Dict[str, IterativeComplexityAnalysis]:
    """API pública para análisis de complejidad"""
    analyzer = BasicComplexityAnalyzer()
    results = {}
    
    for procedure in ast.procedures:
        results[procedure.name] = analyzer.analyze_procedure(procedure)
    
    return results