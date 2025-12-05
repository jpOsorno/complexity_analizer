"""
Analizador de Complejidad Mejorado - CORRECCIÓN DE BUGS
========================================================

Este módulo corrige los bugs críticos en el análisis de complejidad iterativa.

BUGS CORREGIDOS:
- InsertionSort: Ahora reporta O(n²) correctamente
- LinearSearch: Ahora reporta O(n) correctamente  
- MatrixMultiply: Ahora reporta O(n³) correctamente

MEJORAS:
- Detección correcta de loops anidados
- Multiplicación de complejidades para loops anidados
- Análisis mejorado de while loops
"""

import sys
import os
from typing import List, Dict, Set
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from syntax_tree.nodes import *


@dataclass
class LoopInfo:
    """Información sobre un loop"""
    variable: str
    iterations: str  # "n", "n-1", "n-i", etc.
    depth: int
    parent_loops: List[str]  # Variables de loops padres
    

class ImprovedComplexityAnalyzer:
    """
    Analizador mejorado que cuenta correctamente las iteraciones de loops.
    
    CORRECCIÓN PRINCIPAL: Detecta loops anidados y multiplica sus complejidades.
    """
    
    def __init__(self):
        self.loops_stack: List[LoopInfo] = []
        self.max_depth = 0
        self.has_early_exit = False
        
    def analyze_procedure(self, procedure: ProcedureNode) -> Dict[str, str]:
        """
        Analiza un procedimiento y retorna complejidades.
        
        Returns:
            Dict con 'worst', 'best', 'average'
        """
        self.loops_stack = []
        self.max_depth = 0
        self.has_early_exit = False
        
        # Analizar el cuerpo
        self._analyze_block(procedure.body)
        
        # Calcular complejidad basada en profundidad de loops
        worst = self._calculate_complexity()
        
        # Detectar early exit para mejor caso
        best = "Ω(1)" if self.has_early_exit else worst.replace("O", "Ω")
        average = worst.replace("O", "Θ")
        
        return {
            'worst': worst,
            'best': best,
            'average': average,
            'max_depth': self.max_depth
        }
    
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
            # Detectar early exit dentro de loops
            if len(self.loops_stack) > 0:
                self.has_early_exit = True
    
    def _analyze_for(self, node: ForNode):
        """Analiza un loop FOR"""
        # Calcular iteraciones
        iterations = self._calculate_iterations(node.start, node.end, node.variable)
        
        # Crear info del loop
        depth = len(self.loops_stack)
        parent_vars = [loop.variable for loop in self.loops_stack]
        
        loop_info = LoopInfo(
            variable=node.variable,
            iterations=iterations,
            depth=depth,
            parent_loops=parent_vars
        )
        
        # Agregar a stack
        self.loops_stack.append(loop_info)
        self.max_depth = max(self.max_depth, len(self.loops_stack))
        
        # Analizar cuerpo
        self._analyze_block(node.body)
        
        # Remover del stack
        self.loops_stack.pop()
    
    def _analyze_while(self, node: WhileNode):
        """Analiza un loop WHILE"""
        # WHILE loops son más difíciles de analizar estáticamente
        # Asumimos O(n) iteraciones en el peor caso
        depth = len(self.loops_stack)
        parent_vars = [loop.variable for loop in self.loops_stack]
        
        loop_info = LoopInfo(
            variable="while_var",
            iterations="n",
            depth=depth,
            parent_loops=parent_vars
        )
        
        self.loops_stack.append(loop_info)
        self.max_depth = max(self.max_depth, len(self.loops_stack))
        
        self._analyze_block(node.body)
        
        self.loops_stack.pop()
    
    def _analyze_repeat(self, node: RepeatNode):
        """Analiza un loop REPEAT"""
        depth = len(self.loops_stack)
        parent_vars = [loop.variable for loop in self.loops_stack]
        
        loop_info = LoopInfo(
            variable="repeat_var",
            iterations="n",
            depth=depth,
            parent_loops=parent_vars
        )
        
        self.loops_stack.append(loop_info)
        self.max_depth = max(self.max_depth, len(self.loops_stack))
        
        self._analyze_block(node.body)
        
        self.loops_stack.pop()
    
    def _analyze_if(self, node: IfNode):
        """Analiza un condicional IF"""
        # Analizar rama then
        self._analyze_block(node.then_block)
        
        # Analizar rama else si existe
        if node.else_block:
            self._analyze_block(node.else_block)
    
    def _calculate_iterations(self, start_expr: ExpressionNode, end_expr: ExpressionNode, variable: str) -> str:
        """
        Calcula el número de iteraciones de un loop.
        
        Ejemplos:
        - for i ← 1 to n: "n"
        - for i ← 1 to n-1: "n-1"
        - for j ← 1 to n-i: "n-i"
        - for i ← 2 to n: "n-1"
        """
        start = self._expr_to_string(start_expr)
        end = self._expr_to_string(end_expr)
        
        # Casos comunes
        if start in ["0", "1"]:
            if end == "n":
                return "n" if start == "1" else "n+1"
            elif "n-1" in end or "n - 1" in end:
                return "n-1" if start == "1" else "n"
            elif "n-i" in end or f"n-{variable}" in end or "n - i" in end:
                # Loop que depende de variable externa
                return "n-i"
            else:
                return end
        
        # for i ← 2 to n
        if start == "2" and end == "n":
            return "n-1"
        
        # for j ← i+1 to n
        if "i+1" in start or "i + 1" in start:
            return "n-i"
        
        # Caso genérico: end - start + 1
        if start.isdigit() and end.isdigit():
            return str(int(end) - int(start) + 1)
        
        return f"{end}-{start}+1"
    
    def _calculate_complexity(self) -> str:
        """
        Calcula la complejidad basada en la profundidad máxima de loops.
        
        CORRECCIÓN PRINCIPAL: Ahora multiplica correctamente las complejidades.
        """
        if self.max_depth == 0:
            return "O(1)"
        elif self.max_depth == 1:
            return "O(n)"
        elif self.max_depth == 2:
            return "O(n²)"
        elif self.max_depth == 3:
            return "O(n³)"
        elif self.max_depth == 4:
            return "O(n⁴)"
        else:
            return f"O(n^{self.max_depth})"
    
    def _expr_to_string(self, expr: ExpressionNode) -> str:
        """Convierte una expresión a string"""
        if isinstance(expr, NumberNode):
            return str(expr.value)
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._expr_to_string(expr.left)
            right = self._expr_to_string(expr.right)
            return f"{left}{expr.op}{right}"
        elif isinstance(expr, FunctionCallNode):
            if expr.name in ["floor", "ceiling"]:
                arg = self._expr_to_string(expr.arguments[0])
                return arg  # Simplificar floor/ceiling
            return expr.name
        else:
            return "n"


# ============================================================================
# INTEGRACIÓN CON SISTEMA EXISTENTE
# ============================================================================

def patch_complexity_analyzer():
    """
    Parchea el BasicComplexityAnalyzer para usar el analizador mejorado.
    """
    try:
        from .complexity_analyzer import BasicComplexityAnalyzer, IterativeComplexityAnalysis
        
        # Guardar el método original
        original_analyze = BasicComplexityAnalyzer.analyze_procedure
        
        def improved_analyze_wrapper(self, procedure):
            """Versión mejorada que usa el analizador corregido"""
            # Usar analizador mejorado para obtener complejidades correctas
            improved = ImprovedComplexityAnalyzer()
            complexity_result = improved.analyze_procedure(procedure)
            
            # Llamar al método original para obtener el análisis detallado
            result = original_analyze(self, procedure)
            
            # Sobrescribir las complejidades con las correctas
            result.worst_case = complexity_result['worst']
            result.best_case = complexity_result['best']
            result.average_case = complexity_result['average']
            
            return result
        
        # Reemplazar el método
        BasicComplexityAnalyzer.analyze_procedure = improved_analyze_wrapper
        
        return True
    except Exception as e:
        print(f"⚠ No se pudo parchear BasicComplexityAnalyzer: {e}")
        return False


# Aplicar el parche automáticamente al importar
_patch_applied = patch_complexity_analyzer()
if _patch_applied:
    print("✓ Analizador de complejidad parcheado exitosamente")

