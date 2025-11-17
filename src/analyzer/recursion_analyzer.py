"""
Analizador de Recursión
========================

Detecta algoritmos recursivos y genera ecuaciones de recurrencia.

Funcionalidades:
1. Detectar si un procedimiento es recursivo
2. Identificar el tipo de recursión (simple, múltiple, indirecta)
3. Extraer el caso base
4. Extraer las llamadas recursivas
5. Generar la ecuación de recurrencia simbólica

Ejemplo:
    Factorial(n):
        if n ≤ 1: return 1
        else: return n * Factorial(n-1)
    
    Ecuación: T(n) = T(n-1) + O(1)
    Caso base: T(1) = O(1)
"""

import sys
import os
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from sympy import symbols, sympify, simplify, Eq, Function

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..syntax_tree.nodes import *
from analyzer.visitor import ASTVisitor


# ============================================================================
# ESTRUCTURAS DE DATOS PARA RECURSIÓN
# ============================================================================

@dataclass
class RecursiveCall:
    """Representa una llamada recursiva dentro del código"""
    function_name: str
    arguments: List[ExpressionNode]
    depth_reduction: Optional[str] = None  # ej: "n-1", "n/2"
    location: str = ""  # Descripción de dónde ocurre


@dataclass
class RecurrenceEquation:
    """Representa una ecuación de recurrencia"""
    function_name: str
    parameter: str  # Variable principal (ej: "n")
    
    # Caso base
    base_case_condition: str  # ej: "n ≤ 1"
    base_case_cost: str       # ej: "O(1)" o "1"
    
    # Caso recursivo
    recursive_calls: List[RecursiveCall]
    non_recursive_cost: str   # ej: "O(1)", "O(n)"
    
    # Ecuación completa
    equation_str: str = ""    # ej: "T(n) = T(n-1) + O(1)"
    
    # Clasificación
    recursion_type: str = ""  # "linear", "binary", "multiple", "divide-and-conquer"
    
    def __post_init__(self):
        """Genera la ecuación en formato string"""
        if not self.equation_str:
            self.equation_str = self._generate_equation_string()
            self.recursion_type = self._classify_recursion()
    
    def _generate_equation_string(self) -> str:
        """Genera la representación string de la ecuación"""
        if not self.recursive_calls:
            return f"T({self.parameter}) = {self.base_case_cost}"
        
        # Términos recursivos
        recursive_terms = []
        for call in self.recursive_calls:
            if call.depth_reduction:
                recursive_terms.append(f"T({call.depth_reduction})")
            else:
                # Intentar deducir la reducción de los argumentos
                recursive_terms.append("T(...)")
        
        # Ecuación completa
        recursive_part = " + ".join(recursive_terms)
        return f"T({self.parameter}) = {recursive_part} + {self.non_recursive_cost}"
    
    def _classify_recursion(self) -> str:
        """Clasifica el tipo de recursión"""
        num_calls = len(self.recursive_calls)
        
        if num_calls == 0:
            return "non-recursive"
        elif num_calls == 1:
            # Verificar si es lineal o divide-and-conquer
            call = self.recursive_calls[0]
            if call.depth_reduction:
                if "n-1" in call.depth_reduction or "n-" in call.depth_reduction:
                    return "linear"
                elif "n/2" in call.depth_reduction or "n/" in call.depth_reduction:
                    return "divide-and-conquer"
            return "simple"
        elif num_calls == 2:
            return "binary"
        else:
            return "multiple"


@dataclass
class RecursionAnalysisResult:
    """Resultado del análisis de recursión de un procedimiento"""
    procedure_name: str
    is_recursive: bool
    
    # Si es recursivo
    recurrence_equation: Optional[RecurrenceEquation] = None
    
    # Análisis detallado
    base_cases: List[str] = field(default_factory=list)
    recursive_cases: List[str] = field(default_factory=list)
    
    # Metadata
    max_recursion_depth: Optional[int] = None
    calls_count: int = 0


# ============================================================================
# VISITOR PARA ANÁLISIS DE RECURSIÓN
# ============================================================================

class RecursionAnalyzerVisitor(ASTVisitor):
    """
    Visitor que analiza la recursión en un procedimiento.
    
    Detecta:
    - Llamadas recursivas (directas e indirectas)
    - Casos base
    - Reducción del tamaño del problema
    - Costo no recursivo
    """
    
    def __init__(self, procedure_name: str):
        self.procedure_name = procedure_name
        self.current_procedure = procedure_name
        
        # Resultados del análisis
        self.recursive_calls: List[RecursiveCall] = []
        self.base_cases: List[Tuple[str, str]] = []  # (condición, costo)
        
        # Estado durante el recorrido
        self.in_base_case = False
        self.current_condition = None
        self.parameter_name = None  # Variable principal (ej: "n")
    
    def visit_program(self, node: ProgramNode):
        """Visita el programa (no usado directamente)"""
        for proc in node.procedures:
            if proc.name == self.procedure_name:
                return proc.accept(self)
        return None
    
    def visit_procedure(self, node: ProcedureNode):
        """Analiza un procedimiento completo"""
        # Identificar el parámetro principal (asumimos el primero por ahora)
        if node.parameters:
            first_param = node.parameters[0]
            if isinstance(first_param, SimpleParamNode):
                self.parameter_name = first_param.name
            elif isinstance(first_param, ArrayParamNode):
                # Para arrays, el tamaño suele ser el segundo parámetro
                if len(node.parameters) > 1 and isinstance(node.parameters[1], SimpleParamNode):
                    self.parameter_name = node.parameters[1].name
        
        # Analizar el cuerpo
        node.body.accept(self)
        
        return self._build_result()
    
    def visit_if(self, node: IfNode):
        """
        Analiza IF-THEN-ELSE para detectar casos base vs recursivos.
        
        Patrón común:
            if (n ≤ 1) then
                return 1        ← Caso base
            else
                return F(n-1)   ← Caso recursivo
        """
        # Guardar condición actual
        old_condition = self.current_condition
        self.current_condition = self._condition_to_string(node.condition)
        
        # Analizar rama THEN (posible caso base)
        old_base_flag = self.in_base_case
        self.in_base_case = True
        
        # Verificar si THEN tiene return sin recursión
        has_recursive_call_in_then = self._has_recursive_call(node.then_block)
        
        if not has_recursive_call_in_then:
            # Es un caso base
            cost = self._estimate_cost(node.then_block)
            self.base_cases.append((self.current_condition, cost))
        
        node.then_block.accept(self)
        self.in_base_case = old_base_flag
        
        # Analizar rama ELSE (posible caso recursivo)
        if node.else_block:
            self.in_base_case = False
            node.else_block.accept(self)
        
        self.current_condition = old_condition
    
    def visit_for(self, node: ForNode):
        """Analiza FOR (no debería haber recursión aquí, pero contar costo)"""
        node.body.accept(self)
    
    def visit_while(self, node: WhileNode):
        """Analiza WHILE (no debería haber recursión aquí)"""
        node.body.accept(self)
    
    def visit_repeat(self, node: RepeatNode):
        """Analiza REPEAT-UNTIL"""
        node.body.accept(self)
    
    def visit_function_call(self, node: FunctionCallNode):
        """Detecta llamadas recursivas en expresiones"""
        if node.name == self.procedure_name:
            # ¡Llamada recursiva!
            call = self._analyze_recursive_call(node)
            self.recursive_calls.append(call)
        
        # Visitar argumentos por si hay recursión anidada
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_call_statement(self, node: CallStatementNode):
        """Detecta llamadas recursivas como statements"""
        if node.name == self.procedure_name:
            # ¡Llamada recursiva!
            call = self._analyze_recursive_call_statement(node)
            self.recursive_calls.append(call)
        
        # Visitar argumentos
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_return(self, node: ReturnNode):
        """Analiza RETURN (puede tener llamada recursiva)"""
        if node.value:
            node.value.accept(self)
    
    def visit_assignment(self, node: AssignmentNode):
        """Analiza asignaciones (pueden tener llamadas recursivas)"""
        node.value.accept(self)
    
    # ========================================================================
    # MÉTODOS AUXILIARES
    # ========================================================================
    
    def _has_recursive_call(self, block: BlockNode) -> bool:
        """Verifica si un bloque contiene llamadas recursivas"""
        # Visitor temporal para buscar llamadas
        class CallFinder(ASTVisitor):
            def __init__(self, target_name):
                self.target_name = target_name
                self.found = False
            
            def visit_program(self, node): pass
            def visit_procedure(self, node): pass
            def visit_for(self, node): 
                node.body.accept(self)
            def visit_while(self, node):
                node.body.accept(self)
            def visit_repeat(self, node):
                node.body.accept(self)
            def visit_if(self, node):
                node.then_block.accept(self)
                if node.else_block:
                    node.else_block.accept(self)
            
            def visit_function_call(self, node):
                if node.name == self.target_name:
                    self.found = True
            
            def visit_call_statement(self, node):
                if node.name == self.target_name:
                    self.found = True
        
        finder = CallFinder(self.procedure_name)
        block.accept(finder)
        return finder.found
    
    def _condition_to_string(self, condition: ExpressionNode) -> str:
        """Convierte una condición a string legible"""
        if isinstance(condition, BinaryOpNode):
            left = self._expr_to_string(condition.left)
            right = self._expr_to_string(condition.right)
            return f"{left} {condition.op} {right}"
        elif isinstance(condition, UnaryOpNode):
            operand = self._expr_to_string(condition.operand)
            return f"{condition.op} {operand}"
        elif isinstance(condition, IdentifierNode):
            return condition.name
        elif isinstance(condition, BooleanNode):
            return str(condition.value)
        else:
            return "?"
    
    def _expr_to_string(self, expr: ExpressionNode) -> str:
        """Convierte una expresión a string"""
        if isinstance(expr, NumberNode):
            return str(expr.value)
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._expr_to_string(expr.left)
            right = self._expr_to_string(expr.right)
            return f"({left} {expr.op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self._expr_to_string(expr.operand)
            return f"{expr.op}{operand}"
        else:
            return "?"
    
    def _analyze_recursive_call(self, node: FunctionCallNode) -> RecursiveCall:
        """Analiza una llamada recursiva y extrae información"""
        # Extraer el argumento (asumimos que el primero es el tamaño)
        depth_reduction = None
        
        if node.arguments:
            arg = node.arguments[0]
            arg_str = self._expr_to_string(arg)
            
            # Detectar patrones comunes
            if self.parameter_name and self.parameter_name in arg_str:
                depth_reduction = arg_str
        
        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location=f"expression"
        )
    
    def _analyze_recursive_call_statement(self, node: CallStatementNode) -> RecursiveCall:
        """Analiza una llamada recursiva como statement"""
        depth_reduction = None
        
        if node.arguments:
            arg = node.arguments[0]
            arg_str = self._expr_to_string(arg)
            
            if self.parameter_name and self.parameter_name in arg_str:
                depth_reduction = arg_str
        
        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location=f"statement"
        )
    
    def _estimate_cost(self, block: BlockNode) -> str:
        """Estima el costo de un bloque (sin recursión)"""
        # Contar operaciones básicas
        from analyzer.visitor import CountingVisitor
        
        # Por ahora, retornar O(1) si no hay ciclos
        # TODO: Implementar análisis más sofisticado
        return "O(1)"
    
    def _build_result(self) -> RecursionAnalysisResult:
        """Construye el resultado final del análisis"""
        is_recursive = len(self.recursive_calls) > 0
        
        if not is_recursive:
            return RecursionAnalysisResult(
                procedure_name=self.procedure_name,
                is_recursive=False
            )
        
        # Construir ecuación de recurrencia
        if self.base_cases:
            base_condition, base_cost = self.base_cases[0]
        else:
            base_condition = "n ≤ 1"
            base_cost = "O(1)"
        
        recurrence = RecurrenceEquation(
            function_name=self.procedure_name,
            parameter=self.parameter_name or "n",
            base_case_condition=base_condition,
            base_case_cost=base_cost,
            recursive_calls=self.recursive_calls,
            non_recursive_cost="O(1)"  # Por ahora simplificado
        )
        
        return RecursionAnalysisResult(
            procedure_name=self.procedure_name,
            is_recursive=True,
            recurrence_equation=recurrence,
            base_cases=[bc[0] for bc in self.base_cases],
            recursive_cases=[f"{call.function_name}({call.depth_reduction})" 
                           for call in self.recursive_calls],
            calls_count=len(self.recursive_calls)
        )


# ============================================================================
# API PÚBLICA
# ============================================================================

def analyze_recursion(procedure: ProcedureNode) -> RecursionAnalysisResult:
    """
    Analiza un procedimiento para detectar recursión.
    
    Args:
        procedure: Nodo del procedimiento a analizar
        
    Returns:
        RecursionAnalysisResult con el análisis completo
        
    Example:
        >>> ast = parse(factorial_code)
        >>> proc = ast.procedures[0]
        >>> result = analyze_recursion(proc)
        >>> print(result.recurrence_equation.equation_str)
        T(n) = T(n-1) + O(1)
    """
    visitor = RecursionAnalyzerVisitor(procedure.name)
    return visitor.visit_procedure(procedure)


def analyze_all_procedures(ast: ProgramNode) -> Dict[str, RecursionAnalysisResult]:
    """
    Analiza todos los procedimientos de un programa.
    
    Args:
        ast: AST del programa completo
        
    Returns:
        Diccionario {nombre_procedimiento: resultado_análisis}
    """
    results = {}
    
    for procedure in ast.procedures:
        result = analyze_recursion(procedure)
        results[procedure.name] = result
    
    return results


# ============================================================================
# EJEMPLO Y TESTS
# ============================================================================

def demo():
    """Demuestra el análisis de recursión con ejemplos"""
    from parser.parser import parse
    
    examples = {
        "Factorial": """
Factorial(n)
begin
    if (n ≤ 1) then
    begin
        return 1
    end
    else
    begin
        return n * call Factorial(n-1)
    end
end
        """,
        
        "Fibonacci": """
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
        """,
        
        "BinarySearch": """
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
    else
    begin
        if (A[mid] < x) then
        begin
            return call BinarySearch(A, mid+1, right, x)
        end
        else
        begin
            return call BinarySearch(A, left, mid-1, x)
        end
    end
end
        """,
        
        "MergeSort": """
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
        """
    }
    
    print("="*70)
    print("ANÁLISIS DE RECURSIÓN - EJEMPLOS")
    print("="*70)
    
    for name, code in examples.items():
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print(f"{'='*70}")
        print(f"\nCódigo:")
        print(code)
        
        try:
            # Parsear
            ast = parse(code)
            proc = ast.procedures[0]
            
            # Analizar
            result = analyze_recursion(proc)
            
            print(f"\n{'─'*70}")
            print("RESULTADOS:")
            print(f"{'─'*70}")
            
            print(f"\n¿Es recursivo?: {'Sí' if result.is_recursive else 'No'}")
            
            if result.is_recursive and result.recurrence_equation:
                eq = result.recurrence_equation
                
                print(f"\n📐 Ecuación de Recurrencia:")
                print(f"   {eq.equation_str}")
                
                print(f"\n📌 Caso Base:")
                print(f"   Condición: {eq.base_case_condition}")
                print(f"   Costo: {eq.base_case_cost}")
                
                print(f"\n🔄 Llamadas Recursivas:")
                for i, call in enumerate(eq.recursive_calls, 1):
                    print(f"   {i}. {call.function_name}({call.depth_reduction or '...'})")
                
                print(f"\n🏷️  Tipo de Recursión: {eq.recursion_type}")
                print(f"💰 Costo No Recursivo: {eq.non_recursive_cost}")
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()