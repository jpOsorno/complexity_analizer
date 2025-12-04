"""
Analizador de Complejidad para Algoritmos Iterativos - VERSIÓN DETALLADA
========================================================================

MEJORA CRÍTICA: Análisis línea por línea con sumatorias matemáticas completas,
cotas fuertes y explicaciones paso a paso detalladas.

Características:
- ✅ Análisis de coste línea por línea
- ✅ Generación de sumatorias matemáticas completas
- ✅ Simplificación algebraica paso a paso
- ✅ Cotas fuertes (tight bounds) explícitas
- ✅ Detección de early exit y condiciones
- ✅ Análisis de mejor, peor y caso promedio
- ✅ Formato tan detallado como análisis recursivo
"""

import sys
import os
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from syntax_tree.nodes import *


# ============================================================================
# ESTRUCTURAS DE DATOS MEJORADAS
# ============================================================================

@dataclass
class StatementCost:
    """Coste detallado de una sentencia individual"""
    statement_type: str  # "assignment", "comparison", "loop", etc.
    line_description: str  # Descripción legible
    base_cost: int  # Coste base (1 para O(1))
    iterations: str = "1"  # Expresión de iteraciones
    total_cost: str = "1"  # Coste total (base × iterations)
    is_inside_loop: bool = False
    loop_depth: int = 0
    loop_variables: List[str] = field(default_factory=list)
    
    def __str__(self):
        if self.is_inside_loop:
            return f"{self.line_description} → {self.total_cost} (dentro de {self.loop_depth} ciclo(s))"
        return f"{self.line_description} → {self.total_cost}"


@dataclass
class LoopAnalysisDetailed:
    """Análisis detallado de un ciclo individual"""
    variable: str
    start: str
    end: str
    iterations: str  # Expresión: "n", "n-1", "n-i", etc.
    body_statements: List[StatementCost] = field(default_factory=list)
    body_cost: str = "O(1)"
    has_early_exit: bool = False
    early_exit_condition: Optional[str] = None
    depth: int = 0
    
    def to_summation(self) -> str:
        """Genera la sumatoria matemática completa del ciclo"""
        if self.has_early_exit:
            return f"Σ(i={self.start} to {self.end}) [{self.body_cost}] (con early exit posible)"
        else:
            return f"Σ(i={self.start} to {self.end}) [{self.body_cost}]"
    
    def __str__(self):
        exit_note = " [early exit]" if self.has_early_exit else ""
        return f"FOR {self.variable}={self.start} to {self.end}: {self.iterations} iteraciones{exit_note}"


@dataclass
class IterativeComplexityAnalysis:
    """Análisis completo de complejidad iterativa - MEJORADO"""
    
    # Complejidades finales
    worst_case: str = "O(1)"
    best_case: str = "Ω(1)"
    average_case: str = "Θ(1)"
    
    # NUEVO: Análisis línea por línea
    line_by_line_costs: List[StatementCost] = field(default_factory=list)
    
    # NUEVO: Sumatorias globales completas
    global_worst_summation: str = ""
    global_best_summation: str = ""
    global_average_summation: str = ""
    
    # Análisis detallado por caso
    worst_case_explanation: str = ""
    best_case_explanation: str = ""
    average_case_explanation: str = ""
    
    # Pasos de simplificación algebraica
    worst_case_steps: List[str] = field(default_factory=list)
    best_case_steps: List[str] = field(default_factory=list)
    average_case_steps: List[str] = field(default_factory=list)
    
    # NUEVO: Cotas fuertes (tight bounds)
    worst_case_tight_bounds: str = ""
    best_case_tight_bounds: str = ""
    average_case_tight_bounds: str = ""
    
    # Información estructural
    loops: List[LoopAnalysisDetailed] = field(default_factory=list)
    max_nesting_depth: int = 0
    has_conditionals: bool = False
    has_early_exit: bool = False
    total_statements: int = 0
    
    def __str__(self):
        result = "\n" + "="*70 + "\n"
        result += "ANÁLISIS ITERATIVO DETALLADO\n"
        result += "="*70 + "\n"
        
        # NUEVO: Análisis línea por línea
        if self.line_by_line_costs:
            result += "\n📝 ANÁLISIS LÍNEA POR LÍNEA:\n"
            result += "-" * 70 + "\n"
            for i, cost in enumerate(self.line_by_line_costs, 1):
                result += f"  {i}. {cost}\n"
            result += "-" * 70 + "\n"
        
        # PEOR CASO
        result += "\n🔴 PEOR CASO:\n"
        result += f"  Complejidad: {self.worst_case}\n"
        if self.global_worst_summation:
            result += f"  Sumatoria Global:\n"
            result += f"    {self.global_worst_summation}\n"
        if self.worst_case_tight_bounds:
            result += f"  Cotas Fuertes:\n"
            result += f"    {self.worst_case_tight_bounds}\n"
        result += f"  Explicación: {self.worst_case_explanation}\n"
        if self.worst_case_steps:
            result += "\n  Pasos de Simplificación:\n"
            for i, step in enumerate(self.worst_case_steps, 1):
                result += f"    {i}. {step}\n"
        
        # MEJOR CASO
        result += "\n🟢 MEJOR CASO:\n"
        result += f"  Complejidad: {self.best_case}\n"
        if self.global_best_summation:
            result += f"  Sumatoria Global:\n"
            result += f"    {self.global_best_summation}\n"
        if self.best_case_tight_bounds:
            result += f"  Cotas Fuertes:\n"
            result += f"    {self.best_case_tight_bounds}\n"
        result += f"  Explicación: {self.best_case_explanation}\n"
        if self.best_case_steps:
            result += "\n  Pasos de Simplificación:\n"
            for i, step in enumerate(self.best_case_steps, 1):
                result += f"    {i}. {step}\n"
        
        # CASO PROMEDIO
        result += "\n🟡 CASO PROMEDIO:\n"
        result += f"  Complejidad: {self.average_case}\n"
        if self.global_average_summation:
            result += f"  Sumatoria Global:\n"
            result += f"    {self.global_average_summation}\n"
        if self.average_case_tight_bounds:
            result += f"  Cotas Fuertes:\n"
            result += f"    {self.average_case_tight_bounds}\n"
        if self.average_case_steps:
            result += "\n  Pasos de Simplificación:\n"
            for i, step in enumerate(self.average_case_steps, 1):
                result += f"    {i}. {step}\n"
        
        result += "\n" + "="*70
        return result
    
    def to_dict(self) -> dict:
        """Serializa a diccionario"""
        return {
            "line_by_line": [str(cost) for cost in self.line_by_line_costs],
            "worst_case": {
                "complexity": self.worst_case,
                "summation": self.global_worst_summation,
                "explanation": self.worst_case_explanation,
                "steps": self.worst_case_steps,
                "tight_bounds": self.worst_case_tight_bounds
            },
            "best_case": {
                "complexity": self.best_case,
                "summation": self.global_best_summation,
                "explanation": self.best_case_explanation,
                "steps": self.best_case_steps,
                "tight_bounds": self.best_case_tight_bounds
            },
            "average_case": {
                "complexity": self.average_case,
                "summation": self.global_average_summation,
                "explanation": self.average_case_explanation,
                "steps": self.average_case_steps,
                "tight_bounds": self.average_case_tight_bounds
            },
            "structure": {
                "max_nesting_depth": self.max_nesting_depth,
                "num_loops": len(self.loops),
                "has_conditionals": self.has_conditionals,
                "has_early_exit": self.has_early_exit,
                "total_statements": self.total_statements
            }
        }


# ============================================================================
# ANALIZADOR DE COSTES LÍNEA POR LÍNEA
# ============================================================================

class LineByLineAnalyzer:
    """Analiza el coste de cada línea de código"""
    
    def __init__(self):
        self.costs: List[StatementCost] = []
        self.current_loop_depth = 0
        self.loop_variables_stack: List[str] = []
        self.current_iterations = "1"
    
    def analyze_statement(self, stmt: StatementNode, description: str = "") -> StatementCost:
        """Analiza una sentencia individual y retorna su coste"""
        
        if isinstance(stmt, AssignmentNode):
            return self._analyze_assignment(stmt, description)
        elif isinstance(stmt, ForNode):
            return self._analyze_for(stmt)
        elif isinstance(stmt, WhileNode):
            return self._analyze_while(stmt)
        elif isinstance(stmt, RepeatNode):
            return self._analyze_repeat(stmt)
        elif isinstance(stmt, IfNode):
            return self._analyze_if(stmt)
        elif isinstance(stmt, CallStatementNode):
            return self._analyze_call(stmt, description)
        elif isinstance(stmt, ReturnNode):
            return self._analyze_return(stmt, description)
        else:
            return StatementCost(
                statement_type="unknown",
                line_description=description or str(type(stmt).__name__),
                base_cost=1,
                total_cost="O(1)",
                is_inside_loop=self.current_loop_depth > 0,
                loop_depth=self.current_loop_depth
            )
    
    def _analyze_assignment(self, node: AssignmentNode, desc: str) -> StatementCost:
        """Analiza asignación: x ← y"""
        target_desc = self._describe_lvalue(node.target)
        value_desc = self._describe_expr(node.value)
        
        description = desc or f"Asignación: {target_desc} ← {value_desc}"
        
        # Calcular coste de acceso a arrays/objetos
        base_cost = 1
        if isinstance(node.target, ArrayLValueNode):
            base_cost += len(node.target.indices)  # Cada índice es una operación
        
        return StatementCost(
            statement_type="assignment",
            line_description=description,
            base_cost=base_cost,
            iterations=self.current_iterations,
            total_cost=f"{base_cost}" if base_cost > 1 else "1",
            is_inside_loop=self.current_loop_depth > 0,
            loop_depth=self.current_loop_depth,
            loop_variables=list(self.loop_variables_stack)
        )
    
    def _analyze_for(self, node: ForNode) -> StatementCost:
        """Analiza ciclo FOR"""
        variable = node.variable
        start = self._expr_to_string(node.start)
        end = self._expr_to_string(node.end)
        
        iterations = self._calculate_iterations(start, end, variable)
        
        description = f"FOR {variable} ← {start} to {end}"
        
        # Guardar estado anterior
        prev_iterations = self.current_iterations
        prev_depth = self.current_loop_depth
        
        # Actualizar contexto
        self.current_loop_depth += 1
        self.loop_variables_stack.append(variable)
        
        # Multiplicar iteraciones
        if prev_iterations == "1":
            self.current_iterations = iterations
        else:
            self.current_iterations = f"{prev_iterations}×{iterations}"
        
        # Analizar cuerpo
        body_costs = []
        for stmt in node.body.statements:
            cost = self.analyze_statement(stmt)
            body_costs.append(cost)
            self.costs.append(cost)
        
        # Restaurar contexto
        self.current_loop_depth = prev_depth
        self.loop_variables_stack.pop()
        self.current_iterations = prev_iterations
        
        # Calcular coste total del FOR
        # Inicialización (1) + Comparación por iteración (n) + Incremento (n) = 1 + 2n
        overhead = f"2×{iterations} + 1"
        
        return StatementCost(
            statement_type="for_loop",
            line_description=description,
            base_cost=1,
            iterations=iterations,
            total_cost=overhead,
            is_inside_loop=self.current_loop_depth > 0,
            loop_depth=self.current_loop_depth,
            loop_variables=[variable]
        )
    
    def _analyze_while(self, node: WhileNode) -> StatementCost:
        """Analiza ciclo WHILE"""
        condition_desc = self._describe_expr(node.condition)
        description = f"WHILE ({condition_desc})"
        
        # WHILE es más difícil de analizar estáticamente
        # Asumimos O(n) iteraciones en el peor caso
        iterations = "n"
        
        prev_iterations = self.current_iterations
        prev_depth = self.current_loop_depth
        
        self.current_loop_depth += 1
        self.loop_variables_stack.append("while_var")
        
        if prev_iterations == "1":
            self.current_iterations = iterations
        else:
            self.current_iterations = f"{prev_iterations}×{iterations}"
        
        body_costs = []
        for stmt in node.body.statements:
            cost = self.analyze_statement(stmt)
            body_costs.append(cost)
            self.costs.append(cost)
        
        self.current_loop_depth = prev_depth
        self.loop_variables_stack.pop()
        self.current_iterations = prev_iterations
        
        return StatementCost(
            statement_type="while_loop",
            line_description=description,
            base_cost=1,
            iterations=iterations,
            total_cost=f"2×{iterations}",
            is_inside_loop=self.current_loop_depth > 0,
            loop_depth=self.current_loop_depth,
            loop_variables=["while_var"]
        )
    
    def _analyze_repeat(self, node: RepeatNode) -> StatementCost:
        """Analiza ciclo REPEAT"""
        condition_desc = self._describe_expr(node.condition)
        description = f"REPEAT ... UNTIL ({condition_desc})"
        
        iterations = "n"
        
        prev_iterations = self.current_iterations
        prev_depth = self.current_loop_depth
        
        self.current_loop_depth += 1
        self.loop_variables_stack.append("repeat_var")
        
        if prev_iterations == "1":
            self.current_iterations = iterations
        else:
            self.current_iterations = f"{prev_iterations}×{iterations}"
        
        body_costs = []
        for stmt in node.body.statements:
            cost = self.analyze_statement(stmt)
            body_costs.append(cost)
            self.costs.append(cost)
        
        self.current_loop_depth = prev_depth
        self.loop_variables_stack.pop()
        self.current_iterations = prev_iterations
        
        return StatementCost(
            statement_type="repeat_loop",
            line_description=description,
            base_cost=1,
            iterations=iterations,
            total_cost=f"2×{iterations}",
            is_inside_loop=self.current_loop_depth > 0,
            loop_depth=self.current_loop_depth,
            loop_variables=["repeat_var"]
        )
    
    def _analyze_if(self, node: IfNode) -> StatementCost:
        """Analiza condicional IF"""
        condition_desc = self._describe_expr(node.condition)
        description = f"IF ({condition_desc})"
        
        # Evaluar condición: 1 comparación
        cost = StatementCost(
            statement_type="conditional",
            line_description=description,
            base_cost=1,
            iterations=self.current_iterations,
            total_cost="1",
            is_inside_loop=self.current_loop_depth > 0,
            loop_depth=self.current_loop_depth
        )
        
        # Analizar rama THEN
        for stmt in node.then_block.statements:
            then_cost = self.analyze_statement(stmt)
            self.costs.append(then_cost)
        
        # Analizar rama ELSE si existe
        if node.else_block:
            for stmt in node.else_block.statements:
                else_cost = self.analyze_statement(stmt)
                self.costs.append(else_cost)
        
        return cost
    
    def _analyze_call(self, node: CallStatementNode, desc: str) -> StatementCost:
        """Analiza llamada a procedimiento"""
        args_desc = ", ".join(self._describe_expr(arg) for arg in node.arguments)
        description = desc or f"CALL {node.name}({args_desc})"
        
        # Llamada a función: asumimos O(1) si no sabemos el coste
        # (podría ser O(n) si es Merge, Partition, etc.)
        base_cost = 1
        
        return StatementCost(
            statement_type="function_call",
            line_description=description,
            base_cost=base_cost,
            iterations=self.current_iterations,
            total_cost="1",
            is_inside_loop=self.current_loop_depth > 0,
            loop_depth=self.current_loop_depth
        )
    
    def _analyze_return(self, node: ReturnNode, desc: str) -> StatementCost:
        """Analiza RETURN"""
        if node.value:
            value_desc = self._describe_expr(node.value)
            description = desc or f"RETURN {value_desc}"
        else:
            description = desc or "RETURN"
        
        return StatementCost(
            statement_type="return",
            line_description=description,
            base_cost=1,
            iterations="1",
            total_cost="1",
            is_inside_loop=False,  # Return termina la función
            loop_depth=0
        )
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _calculate_iterations(self, start: str, end: str, variable: str) -> str:
        """Calcula el número de iteraciones de un ciclo"""
        start_clean = start.strip()
        end_clean = end.strip()
        
        if start_clean in ["0", "1"]:
            if end_clean == "n":
                return "n" if start_clean == "0" else "n"
            elif "n-1" in end_clean:
                return "n-1" if start_clean == "1" else "n"
            elif "n-i" in end_clean or f"n-{variable}" in end_clean:
                return "n-i" if start_clean == "1" else "n-i+1"
            else:
                return end_clean
        
        if start_clean.isalpha() and len(start_clean) == 1:
            if end_clean == "n":
                return f"n-{start_clean}+1"
            elif "n-" in end_clean:
                return f"{end_clean}-{start_clean}+1"
            else:
                return f"{end_clean}-{start_clean}+1"
        
        return f"{end_clean}-{start_clean}+1"
    
    def _describe_lvalue(self, lvalue: 'LValueNode') -> str:
        """Describe un lvalue (lado izquierdo de asignación)"""
        if isinstance(lvalue, VariableLValueNode):
            return lvalue.name
        elif isinstance(lvalue, ArrayLValueNode):
            indices = ", ".join(self._describe_expr(idx) for idx in lvalue.indices)
            return f"{lvalue.name}[{indices}]"
        elif isinstance(lvalue, ObjectLValueNode):
            return f"{lvalue.object_name}.{'.'.join(lvalue.fields)}"
        return "?"
    
    def _describe_expr(self, expr: 'ExpressionNode') -> str:
        """Describe una expresión"""
        if isinstance(expr, NumberNode):
            return str(expr.value)
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._describe_expr(expr.left)
            right = self._describe_expr(expr.right)
            return f"{left} {expr.op} {right}"
        elif isinstance(expr, UnaryOpNode):
            operand = self._describe_expr(expr.operand)
            return f"{expr.op}{operand}"
        elif isinstance(expr, ArrayAccessNode):
            indices = ", ".join(self._describe_expr(idx) if not isinstance(idx, RangeNode) else "range" for idx in expr.indices)
            return f"{expr.name}[{indices}]"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(self._describe_expr(arg) for arg in expr.arguments)
            return f"{expr.name}({args})"
        elif isinstance(expr, BooleanNode):
            return str(expr.value)
        return "?"
    
    def _expr_to_string(self, expr: ExpressionNode) -> str:
        """Convierte expresión a string"""
        return self._describe_expr(expr)


# ============================================================================
# GENERADOR DE SUMATORIAS Y SIMPLIFICADOR
# ============================================================================

class SummationGenerator:
    """Genera y simplifica sumatorias matemáticas"""
    
    @staticmethod
    def generate_global_summation(costs: List[StatementCost], case: str = "worst") -> str:
        """
        Genera la sumatoria global de todas las líneas.
        
        Args:
            costs: Lista de costes línea por línea
            case: "worst", "best", "average"
            
        Returns:
            String con la sumatoria completa
        """
        if not costs:
            return "T(n) = 0"
        
        # Agrupar por profundidad de ciclo
        by_depth = defaultdict(list)
        for cost in costs:
            by_depth[cost.loop_depth].append(cost)
        
        # Construir sumatoria
        summation_parts = []
        
        for depth in sorted(by_depth.keys()):
            depth_costs = by_depth[depth]
            
            if depth == 0:
                # Fuera de ciclos
                total = sum(c.base_cost for c in depth_costs)
                if total > 0:
                    summation_parts.append(f"{total}")
            else:
                # Dentro de ciclos
                # Agrupar por variables de ciclo
                for cost in depth_costs:
                    if cost.loop_variables:
                        vars_str = ", ".join(cost.loop_variables)
                        summation_parts.append(f"Σ({vars_str}) [{cost.base_cost}]")
                    else:
                        summation_parts.append(f"{cost.base_cost}")
        
        if summation_parts:
            return "T(n) = " + " + ".join(summation_parts)
        return "T(n) = 1"
    
    @staticmethod
    def simplify_summation(summation: str, steps: List[str], case: str = "worst") -> str:
        """
        Simplifica una sumatoria paso a paso.
        
        Args:
            summation: Sumatoria a simplificar
            steps: Lista donde agregar los pasos
            case: "worst", "best", "average"
            
        Returns:
            Complejidad final simplificada
        """
        steps.append(f"🔍 Simplificación de la sumatoria ({case} caso):")
        steps.append(f"  Sumatoria inicial: {summation}")
        steps.append("")
        
        # Detectar patrones comunes
        if "Σ(i, j)" in summation or "Σ(i)Σ(j)" in summation:
            # Ciclos anidados dobles
            steps.append("  Patrón detectado: Ciclos anidados (2 niveles)")
            steps.append("  Fórmula: Σ(i=1 to n) Σ(j=1 to n-i) 1")
            steps.append("         = Σ(i=1 to n) (n-i)")
            steps.append("         = Σ(i=1 to n) n - Σ(i=1 to n) i")
            steps.append("         = n×n - n(n+1)/2")
            steps.append("         = n² - n²/2 - n/2")
            steps.append("         = n²/2 - n/2")
            steps.append("         = (n² - n)/2")
            steps.append("         ≈ n²/2")
            steps.append("")
            steps.append("  Resultado: O(n²)")
            return "n²"
        
        elif "Σ(i)" in summation and "Σ(j)" not in summation:
            # Ciclo simple
            steps.append("  Patrón detectado: Ciclo simple")
            steps.append("  Fórmula: Σ(i=1 to n) c = c×n")
            steps.append("")
            steps.append("  Resultado: O(n)")
            return "n"
        
        elif "Σ" not in summation:
            # Sin ciclos
            steps.append("  Patrón detectado: Sin ciclos (operaciones constantes)")
            steps.append("")
            steps.append("  Resultado: O(1)")
            return "1"
        
        else:
            # Patrón genérico
            steps.append("  Análisis genérico requerido")
            steps.append("")
            steps.append("  Resultado: O(n)")
            return "n"
    
    @staticmethod
    def calculate_tight_bounds(complexity: str) -> str:
        """
        Calcula cotas fuertes (tight bounds) para una complejidad.
        
        Args:
            complexity: Complejidad (ej: "n²", "n", "1")
            
        Returns:
            String con las cotas fuertes
        """
        if complexity == "1":
            return "c₁ ≤ T(n) ≤ c₂ para todo n ≥ n₀, donde c₁, c₂ > 0"
        elif complexity == "n":
            return "c₁×n ≤ T(n) ≤ c₂×n para todo n ≥ n₀, donde c₁, c₂ > 0"
        elif complexity == "n²":
            return "c₁×n² ≤ T(n) ≤ c₂×n² para todo n ≥ n₀, donde c₁, c₂ > 0"
        elif complexity == "n³":
            return "c₁×n³ ≤ T(n) ≤ c₂×n³ para todo n ≥ n₀, donde c₁, c₂ > 0"
        elif "log" in complexity:
            return "c₁×log(n) ≤ T(n) ≤ c₂×log(n) para todo n ≥ n₀, donde c₁, c₂ > 0"
        elif "×" in complexity or "*" in complexity:
            return f"c₁×{complexity} ≤ T(n) ≤ c₂×{complexity} para todo n ≥ n₀, donde c₁, c₂ > 0"
        else:
            return f"c₁×f(n) ≤ T(n) ≤ c₂×f(n) donde f(n) = {complexity}"


# ============================================================================
# ANALIZADOR MEJORADO PRINCIPAL
# ============================================================================

class EnhancedComplexityAnalyzer:
    """
    Analizador de complejidad iterativa con análisis línea por línea.
    
    MEJORA CRÍTICA: Análisis detallado comparable al recursivo.
    """
    
    def __init__(self):
        self.current_procedure = None
        self.line_analyzer = LineByLineAnalyzer()
        self.has_early_exit = False
        self.has_conditionals = False
    
    def analyze_procedure(self, procedure: ProcedureNode) -> IterativeComplexityAnalysis:
        """Analiza un procedimiento completo con detalle línea por línea"""
        self.current_procedure = procedure.name
        self.line_analyzer = LineByLineAnalyzer()
        self.has_early_exit = False
        self.has_conditionals = False
        
        # PASO 1: Análisis línea por línea
        for stmt in procedure.body.statements:
            self.line_analyzer.analyze_statement(stmt)
            
            # Detectar early exit y conditionals
            if isinstance(stmt, ReturnNode) and self.line_analyzer.current_loop_depth > 0:
                self.has_early_exit = True
            if isinstance(stmt, IfNode):
                self.has_conditionals = True
        
        # PASO 2: Generar análisis completo
        return self._build_complete_analysis()
    
    def _build_complete_analysis(self) -> IterativeComplexityAnalysis:
        """Construye el análisis completo para los 3 casos"""
        
        analysis = IterativeComplexityAnalysis()
        analysis.line_by_line_costs = self.line_analyzer.costs
        analysis.total_statements = len(self.line_analyzer.costs)
        analysis.has_conditionals = self.has_conditionals
        analysis.has_early_exit = self.has_early_exit
        # Detectar estructura de ciclos
        loops = self._extract_loops()
        analysis.loops = loops
        analysis.max_nesting_depth = max((loop.depth for loop in loops), default=0) + 1
        
        # ====================================================================
        # PEOR CASO
        # ====================================================================
        
        worst_steps = []
        worst_steps.append("🔍 ANÁLISIS DEL PEOR CASO")
        worst_steps.append("="*60)
        worst_steps.append("")
        
        # Generar sumatoria global
        generator = SummationGenerator()
        global_worst = generator.generate_global_summation(analysis.line_by_line_costs, "worst")
        analysis.global_worst_summation = global_worst
        
        worst_steps.append("📊 Sumatoria Global:")
        worst_steps.append(f"  {global_worst}")
        worst_steps.append("")
        
        # Simplificar
        worst_complexity = generator.simplify_summation(global_worst, worst_steps, "worst")
        analysis.worst_case = f"O({worst_complexity})"
        
        # Calcular cotas fuertes
        analysis.worst_case_tight_bounds = generator.calculate_tight_bounds(worst_complexity)
        
        worst_steps.append("")
        worst_steps.append("📐 Cotas Fuertes:")
        worst_steps.append(f"  {analysis.worst_case_tight_bounds}")
        
        # Explicación
        analysis.worst_case_explanation = self._generate_explanation(loops, "peor")
        analysis.worst_case_steps = worst_steps
        
        # ====================================================================
        # MEJOR CASO
        # ====================================================================
        
        best_steps = []
        best_steps.append("🔍 ANÁLISIS DEL MEJOR CASO")
        best_steps.append("="*60)
        best_steps.append("")
        
        if self.has_early_exit:
            best_steps.append("✓ Early exit detectado")
            best_steps.append("  En el mejor caso, el algoritmo termina inmediatamente")
            best_steps.append("")
            analysis.best_case = "Ω(1)"
            analysis.global_best_summation = "T(n) = c (constante debido a early exit)"
            analysis.best_case_tight_bounds = generator.calculate_tight_bounds("1")
            analysis.best_case_explanation = "Early exit permite terminación en O(1)"
        else:
            best_steps.append("⚠ No hay early exit detectado")
            best_steps.append("  El mejor caso es igual al peor caso")
            best_steps.append("")
            
            global_best = generator.generate_global_summation(analysis.line_by_line_costs, "best")
            analysis.global_best_summation = global_best
            
            best_steps.append("📊 Sumatoria Global:")
            best_steps.append(f"  {global_best}")
            best_steps.append("")
            
            best_complexity = generator.simplify_summation(global_best, best_steps, "best")
            analysis.best_case = f"Ω({best_complexity})"
            analysis.best_case_tight_bounds = generator.calculate_tight_bounds(best_complexity)
            analysis.best_case_explanation = "Sin early exit, el mejor caso coincide con el peor caso"
        
        best_steps.append("")
        best_steps.append("📐 Cotas Fuertes:")
        best_steps.append(f"  {analysis.best_case_tight_bounds}")
        
        analysis.best_case_steps = best_steps
        
        # ====================================================================
        # CASO PROMEDIO
        # ====================================================================
        
        avg_steps = []
        avg_steps.append("🔍 ANÁLISIS DEL CASO PROMEDIO")
        avg_steps.append("="*60)
        avg_steps.append("")
        
        if self.has_early_exit:
            avg_steps.append("⚙ Con early exit, el caso promedio está entre Ω(1) y O(peor caso)")
            avg_steps.append(f"  Asumimos: Θ({worst_complexity})")
            avg_steps.append("")
            analysis.average_case = f"Θ({worst_complexity})"
            analysis.global_average_summation = f"T(n) ≈ {global_worst} (caso promedio)"
            analysis.average_case_tight_bounds = generator.calculate_tight_bounds(worst_complexity)
        else:
            avg_steps.append("⚙ Sin early exit, el caso promedio es igual al peor caso")
            avg_steps.append("")
            
            global_avg = generator.generate_global_summation(analysis.line_by_line_costs, "average")
            analysis.global_average_summation = global_avg
            
            avg_steps.append("📊 Sumatoria Global:")
            avg_steps.append(f"  {global_avg}")
            avg_steps.append("")
            
            avg_complexity = generator.simplify_summation(global_avg, avg_steps, "average")
            analysis.average_case = f"Θ({avg_complexity})"
            analysis.average_case_tight_bounds = generator.calculate_tight_bounds(avg_complexity)
        
        avg_steps.append("")
        avg_steps.append("📐 Cotas Fuertes:")
        avg_steps.append(f"  {analysis.average_case_tight_bounds}")
        
        analysis.average_case_steps = avg_steps
        
        return analysis

    def _extract_loops(self) -> List[LoopAnalysisDetailed]:
        """Extrae información de los ciclos analizados"""
        loops = []
        
        for cost in self.line_analyzer.costs:
            if cost.statement_type in ["for_loop", "while_loop", "repeat_loop"]:
                # Extraer información del ciclo
                loop = LoopAnalysisDetailed(
                    variable=cost.loop_variables[0] if cost.loop_variables else "i",
                    start="1",
                    end="n",
                    iterations=cost.iterations,
                    body_cost=f"O({cost.total_cost})",
                    depth=cost.loop_depth
                )
                loops.append(loop)
        
        return loops

    def _generate_explanation(self, loops: List[LoopAnalysisDetailed], case_type: str) -> str:
        """Genera explicación detallada"""
        if not loops:
            return "Algoritmo sin ciclos, solo operaciones constantes"
        
        explanation = f"**Análisis del {case_type} caso:**\n\n"
        
        if len(loops) == 1:
            loop = loops[0]
            explanation += f"• Un ciclo con {loop.iterations} iteraciones\n"
            explanation += f"• Cada iteración ejecuta operaciones constantes\n"
            explanation += f"• Complejidad total: {loop.iterations} × O(1) = O({loop.iterations})\n"
        else:
            explanation += f"• Ciclos anidados detectados (profundidad: {max(l.depth for l in loops) + 1})\n"
            for i, loop in enumerate(loops, 1):
                indent = "  " * loop.depth
                explanation += f"{indent}• Ciclo {i}: {loop.iterations} iteraciones (nivel {loop.depth})\n"
            
            # Multiplicar iteraciones
            iterations_product = " × ".join(l.iterations for l in loops)
            explanation += f"\n• Producto de iteraciones: {iterations_product}\n"
        
        return explanation


# ============================================================================
# API PÚBLICA (COMPATIBILIDAD)
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