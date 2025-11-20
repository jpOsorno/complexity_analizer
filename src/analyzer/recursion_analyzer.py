"""
Analizador de Recursión - VERSIÓN CORREGIDA
===========================================

Correcciones:
1. Normalización de expresiones (eliminar espacios)
2. Función to_recurrence() para obtener solo la ecuación
3. Mejor detección de reducción de profundidad
"""

import sys
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from syntax_tree.nodes import *
from analyzer.visitor import ASTVisitor


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class RecursiveCall:
    """Representa una llamada recursiva"""
    function_name: str
    arguments: List[ExpressionNode]
    depth_reduction: Optional[str] = None
    location: str = ""


@dataclass
class RecurrenceEquation:
    """Representa una ecuación de recurrencia"""
    function_name: str
    parameter: str
    base_case_condition: str
    base_case_cost: str
    recursive_calls: List[RecursiveCall]
    non_recursive_cost: str
    equation_str: str = ""
    recursion_type: str = ""
    
    def __post_init__(self):
        if not self.equation_str:
            self.equation_str = self._generate_equation_string()
            self.recursion_type = self._classify_recursion()
    
    def _generate_equation_string(self) -> str:
        """Genera la ecuación en formato string"""
        if not self.recursive_calls:
            return f"T({self.parameter}) = {self.base_case_cost}"
        
        # Términos recursivos
        recursive_terms = []
        for call in self.recursive_calls:
            if call.depth_reduction:
                # Normalizar: remover espacios y paréntesis externos
                reduction = call.depth_reduction.replace(" ", "")
                reduction = reduction.strip("()")
                recursive_terms.append(f"T({reduction})")
            else:
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
            call = self.recursive_calls[0]
            if call.depth_reduction:
                reduction = call.depth_reduction.replace(" ", "")
                if "n-1" in reduction or "-1" in reduction:
                    return "linear"
                elif "n/2" in reduction or "/2" in reduction:
                    return "divide-and-conquer"
            return "simple"
        elif num_calls == 2:
            return "binary"
        else:
            return "multiple"
    
    def to_string(self) -> str:
        """Retorna solo la ecuación como string"""
        return self.equation_str
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para fácil serialización"""
        return {
            "function_name": self.function_name,
            "parameter": self.parameter,
            "equation": self.equation_str,
            "type": self.recursion_type,
            "base_case": {
                "condition": self.base_case_condition,
                "cost": self.base_case_cost
            },
            "recursive_cost": self.non_recursive_cost,
            "num_recursive_calls": len(self.recursive_calls)
        }


@dataclass
class RecursionAnalysisResult:
    """Resultado del análisis de recursión"""
    procedure_name: str
    is_recursive: bool
    recurrence_equation: Optional[RecurrenceEquation] = None
    base_cases: List[str] = field(default_factory=list)
    recursive_cases: List[str] = field(default_factory=list)
    max_recursion_depth: Optional[int] = None
    calls_count: int = 0
    
    def to_recurrence(self) -> Optional[str]:
        """
        Retorna SOLO la ecuación de recurrencia como string.
        
        Returns:
            String con la ecuación (ej: "T(n) = T(n-1) + O(1)")
            o None si no es recursivo
        """
        if not self.is_recursive or not self.recurrence_equation:
            return None
        return self.recurrence_equation.to_string()


# ============================================================================
# VISITOR MEJORADO
# ============================================================================

class RecursionAnalyzerVisitor(ASTVisitor):
    """Visitor que analiza recursión"""
    
    def __init__(self, procedure_name: str):
        self.procedure_name = procedure_name
        self.current_procedure = procedure_name
        self.recursive_calls: List[RecursiveCall] = []
        self.base_cases: List[Tuple[str, str]] = []
        self.in_base_case = False
        self.current_condition = None
        self.parameter_name = None
    
    def visit_program(self, node: ProgramNode):
        for proc in node.procedures:
            if proc.name == self.procedure_name:
                return proc.accept(self)
        return None
    
    def visit_procedure(self, node: ProcedureNode):
        # Identificar parámetro principal
        if node.parameters:
            first_param = node.parameters[0]
            if isinstance(first_param, SimpleParamNode):
                self.parameter_name = first_param.name
            elif isinstance(first_param, ArrayParamNode):
                if len(node.parameters) > 1 and isinstance(node.parameters[1], SimpleParamNode):
                    self.parameter_name = node.parameters[1].name
        
        node.body.accept(self)
        return self._build_result()
    
    def visit_if(self, node: IfNode):
        old_condition = self.current_condition
        self.current_condition = self._condition_to_string(node.condition)
        
        old_base_flag = self.in_base_case
        self.in_base_case = True
        
        has_recursive_call_in_then = self._has_recursive_call(node.then_block)
        
        if not has_recursive_call_in_then:
            cost = self._estimate_cost(node.then_block)
            self.base_cases.append((self.current_condition, cost))
        
        node.then_block.accept(self)
        self.in_base_case = old_base_flag
        
        if node.else_block:
            self.in_base_case = False
            node.else_block.accept(self)
        
        self.current_condition = old_condition
    
    def visit_for(self, node: ForNode):
        node.body.accept(self)
    
    def visit_while(self, node: WhileNode):
        node.body.accept(self)
    
    def visit_repeat(self, node: RepeatNode):
        node.body.accept(self)
    
    def visit_function_call(self, node: FunctionCallNode):
        if node.name == self.procedure_name:
            call = self._analyze_recursive_call(node)
            self.recursive_calls.append(call)
        
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_call_statement(self, node: CallStatementNode):
        if node.name == self.procedure_name:
            call = self._analyze_recursive_call_statement(node)
            self.recursive_calls.append(call)
        
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_return(self, node: ReturnNode):
        if node.value:
            node.value.accept(self)
    
    def visit_assignment(self, node: AssignmentNode):
        node.value.accept(self)
    
    # ========================================================================
    # MÉTODOS AUXILIARES MEJORADOS
    # ========================================================================
    
    def _has_recursive_call(self, block: BlockNode) -> bool:
        """Verifica si un bloque contiene llamadas recursivas"""
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
        """Convierte condición a string normalizado"""
        result = self._expr_to_string(condition)
        # Normalizar: eliminar espacios extra
        return " ".join(result.split())
    
    def _expr_to_string(self, expr: ExpressionNode) -> str:
        """Convierte expresión a string (SIN espacios extra)"""
        if isinstance(expr, NumberNode):
            return str(expr.value)
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._expr_to_string(expr.left)
            right = self._expr_to_string(expr.right)
            # SIN espacios alrededor del operador
            return f"{left}{expr.op}{right}"
        elif isinstance(expr, UnaryOpNode):
            operand = self._expr_to_string(expr.operand)
            return f"{expr.op}{operand}"
        elif isinstance(expr, BooleanNode):
            return str(expr.value)
        else:
            return "?"
    
    def _analyze_recursive_call(self, node: FunctionCallNode) -> RecursiveCall:
        """Analiza llamada recursiva"""
        depth_reduction = None
        
        if node.arguments:
            arg = node.arguments[0]
            arg_str = self._expr_to_string(arg)
            
            # Normalizar
            arg_str = arg_str.replace(" ", "")
            
            if self.parameter_name and self.parameter_name in arg_str:
                depth_reduction = arg_str
        
        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location="expression"
        )
    
    def _analyze_recursive_call_statement(self, node: CallStatementNode) -> RecursiveCall:
        """Analiza llamada recursiva como statement"""
        depth_reduction = None
        
        if node.arguments:
            arg = node.arguments[0]
            arg_str = self._expr_to_string(arg)
            arg_str = arg_str.replace(" ", "")
            
            if self.parameter_name and self.parameter_name in arg_str:
                depth_reduction = arg_str
        
        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location="statement"
        )
    
    def _estimate_cost(self, block: BlockNode) -> str:
        """Estima costo del caso base"""
        return "O(1)"
    
    def _build_result(self) -> RecursionAnalysisResult:
        """Construye resultado final"""
        is_recursive = len(self.recursive_calls) > 0
        
        if not is_recursive:
            return RecursionAnalysisResult(
                procedure_name=self.procedure_name,
                is_recursive=False
            )
        
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
            non_recursive_cost="O(1)"
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
    
    Returns:
        RecursionAnalysisResult con el análisis completo
    """
    visitor = RecursionAnalyzerVisitor(procedure.name)
    return visitor.visit_procedure(procedure)


def to_recurrence(code: str, procedure_name: Optional[str] = None) -> Optional[str]:
    """
    Función de conveniencia: extrae SOLO la ecuación de recurrencia.
    
    Args:
        code: Código pseudocódigo
        procedure_name: Nombre del procedimiento (usa el primero si es None)
        
    Returns:
        String con la ecuación (ej: "T(n) = T(n-1) + O(1)")
        o None si no es recursivo
        
    Example:
        >>> code = '''
        ... Factorial(n)
        ... begin
        ...     if (n ≤ 1) then return 1
        ...     else return n * call Factorial(n-1)
        ... end
        ... '''
        >>> print(to_recurrence(code))
        T(n) = T(n-1) + O(1)
    """
    from parser.parser import parse
    
    try:
        ast = parse(code)
        
        if not ast.procedures:
            return None
        
        # Usar el procedimiento especificado o el primero
        if procedure_name:
            proc = next((p for p in ast.procedures if p.name == procedure_name), None)
            if not proc:
                return None
        else:
            proc = ast.procedures[0]
        
        result = analyze_recursion(proc)
        return result.to_recurrence()
    
    except Exception:
        return None


def analyze_all_procedures(ast: ProgramNode) -> Dict[str, RecursionAnalysisResult]:
    """Analiza todos los procedimientos"""
    results = {}
    for procedure in ast.procedures:
        result = analyze_recursion(procedure)
        results[procedure.name] = result
    return results


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demuestra el análisis de recursión"""
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
    print("ANÁLISIS DE RECURSIÓN - DEMO")
    print("="*70)
    
    for name, code in examples.items():
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print(f"{'='*70}")
        
        try:
            # Método 1: Análisis completo
            ast = parse(code)
            proc = ast.procedures[0]
            result = analyze_recursion(proc)
            
            print(f"\n¿Es recursivo?: {'Sí' if result.is_recursive else 'No'}")
            
            if result.is_recursive:
                print(f"\n📐 Ecuación de Recurrencia:")
                print(f"   {result.to_recurrence()}")
                print(f"\n🏷️  Tipo: {result.recurrence_equation.recursion_type}")
            
            # Método 2: Solo ecuación
            print(f"\n💡 Usando to_recurrence():")
            equation = to_recurrence(code)
            print(f"   {equation}")
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()