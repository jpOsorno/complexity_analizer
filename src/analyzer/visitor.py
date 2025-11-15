"""
Visitor Base para Análisis de Complejidad
=========================================

Este módulo implementa el patrón Visitor para recorrer el AST
y extraer información relevante para análisis de complejidad.

Jerarquía de Visitors:
- ASTVisitor: Clase base abstracta con métodos visit_* para cada nodo
- ComplexityVisitor: Calcula complejidad temporal
- SpaceVisitor: Calcula complejidad espacial (futuro)
- LoopAnalyzer: Detecta y analiza ciclos
- RecursionAnalyzer: Detecta y analiza recursión

El patrón Visitor separa el algoritmo de análisis de la estructura del AST,
permitiendo agregar nuevos análisis sin modificar las clases de nodos.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

# Importar nodos del AST
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.syntax_tree.nodes import *


# ============================================================================
# VISITOR BASE ABSTRACTO
# ============================================================================

class ASTVisitor(ABC):
    """
    Clase base abstracta para todos los visitors del AST.
    
    Implementa el patrón Visitor clásico: cada nodo tiene un método visit_*
    correspondiente. Los visitors concretos heredan de esta clase y sobrescriben
    los métodos que les interesan.
    
    Métodos obligatorios:
    - visit_program: Punto de entrada
    - visit_procedure: Analizar un procedimiento
    
    Métodos opcionales: todos los demás (tienen implementación por defecto)
    """
    
    # ========================================================================
    # PUNTO DE ENTRADA
    # ========================================================================
    
    @abstractmethod
    def visit_program(self, node: ProgramNode) -> Any:
        """Visita el nodo raíz del programa"""
        pass
    
    # ========================================================================
    # DEFINICIONES
    # ========================================================================
    
    def visit_class_def(self, node: ClassDefNode) -> Any:
        """Visita una definición de clase"""
        # Por defecto, no hace nada (clases no afectan complejidad temporal)
        return None
    
    @abstractmethod
    def visit_procedure(self, node: ProcedureNode) -> Any:
        """Visita un procedimiento/función"""
        pass
    
    # ========================================================================
    # PARÁMETROS
    # ========================================================================
    
    def visit_simple_param(self, node: SimpleParamNode) -> Any:
        """Visita un parámetro simple"""
        return node.name
    
    def visit_array_param(self, node: ArrayParamNode) -> Any:
        """Visita un parámetro array"""
        return node.name
    
    def visit_class_param(self, node: ClassParamNode) -> Any:
        """Visita un parámetro de tipo clase"""
        return node.name
    
    # ========================================================================
    # BLOQUES Y DECLARACIONES
    # ========================================================================
    
    def visit_block(self, node: BlockNode) -> Any:
        """Visita un bloque de código"""
        # Visitar declaraciones
        for decl in node.declarations:
            decl.accept(self)
        
        # Visitar statements
        results = []
        for stmt in node.statements:
            result = stmt.accept(self)
            if result is not None:
                results.append(result)
        
        return results
    
    def visit_array_decl(self, node: ArrayDeclNode) -> Any:
        """Visita una declaración de array local"""
        # Arrays locales pueden afectar complejidad espacial
        return None
    
    def visit_object_decl(self, node: ObjectDeclNode) -> Any:
        """Visita una declaración de objeto local"""
        return None
    
    # ========================================================================
    # SENTENCIAS (STATEMENTS)
    # ========================================================================
    
    @abstractmethod
    def visit_for(self, node: ForNode) -> Any:
        """Visita un ciclo FOR"""
        pass
    
    @abstractmethod
    def visit_while(self, node: WhileNode) -> Any:
        """Visita un ciclo WHILE"""
        pass
    
    @abstractmethod
    def visit_repeat(self, node: RepeatNode) -> Any:
        """Visita un ciclo REPEAT-UNTIL"""
        pass
    
    @abstractmethod
    def visit_if(self, node: IfNode) -> Any:
        """Visita un condicional IF"""
        pass
    
    def visit_assignment(self, node: AssignmentNode) -> Any:
        """Visita una asignación"""
        # Por defecto, visitar subexpresiones
        node.target.accept(self)
        node.value.accept(self)
        return None
    
    def visit_call_statement(self, node: CallStatementNode) -> Any:
        """Visita una llamada a procedimiento"""
        # Visitar argumentos
        for arg in node.arguments:
            arg.accept(self)
        return None
    
    def visit_return(self, node: ReturnNode) -> Any:
        """Visita un RETURN"""
        if node.value:
            node.value.accept(self)
        return None
    
    # ========================================================================
    # LVALUES
    # ========================================================================
    
    def visit_variable_lvalue(self, node: VariableLValueNode) -> Any:
        """Visita un lvalue de variable simple"""
        return node.name
    
    def visit_array_lvalue(self, node: ArrayLValueNode) -> Any:
        """Visita un lvalue de array"""
        # Visitar índices (pueden ser expresiones complejas)
        for index in node.indices:
            index.accept(self)
        return node.name
    
    def visit_object_lvalue(self, node: ObjectLValueNode) -> Any:
        """Visita un lvalue de objeto"""
        return f"{node.object_name}.{'.'.join(node.fields)}"
    
    # ========================================================================
    # EXPRESIONES
    # ========================================================================
    
    def visit_binary_op(self, node: BinaryOpNode) -> Any:
        """Visita una operación binaria"""
        node.left.accept(self)
        node.right.accept(self)
        return None
    
    def visit_unary_op(self, node: UnaryOpNode) -> Any:
        """Visita una operación unaria"""
        node.operand.accept(self)
        return None
    
    def visit_number(self, node: NumberNode) -> Any:
        """Visita un literal numérico"""
        return node.value
    
    def visit_string(self, node: StringNode) -> Any:
        """Visita un literal string"""
        return node.value
    
    def visit_boolean(self, node: BooleanNode) -> Any:
        """Visita un literal booleano"""
        return node.value
    
    def visit_null(self, node: NullNode) -> Any:
        """Visita NULL"""
        return None
    
    def visit_identifier(self, node: IdentifierNode) -> Any:
        """Visita un identificador"""
        return node.name
    
    def visit_array_access(self, node: ArrayAccessNode) -> Any:
        """Visita un acceso a array"""
        # Visitar índices
        for index in node.indices:
            if isinstance(index, RangeNode):
                index.accept(self)
            else:
                index.accept(self)
        return node.name
    
    def visit_range(self, node: RangeNode) -> Any:
        """Visita un rango (1..n)"""
        node.start.accept(self)
        node.end.accept(self)
        return None
    
    def visit_object_access(self, node: ObjectAccessNode) -> Any:
        """Visita un acceso a campo de objeto"""
        return f"{node.object_name}.{'.'.join(node.fields)}"
    
    def visit_function_call(self, node: FunctionCallNode) -> Any:
        """Visita una llamada a función"""
        for arg in node.arguments:
            arg.accept(self)
        return None
    
    def visit_length(self, node: LengthNode) -> Any:
        """Visita la función length()"""
        return node.array_name
    
    def visit_ceiling(self, node: CeilingNode) -> Any:
        """Visita la función ceiling"""
        node.expression.accept(self)
        return None
    
    def visit_floor(self, node: FloorNode) -> Any:
        """Visita la función floor"""
        node.expression.accept(self)
        return None


# ============================================================================
# VISITOR DE CONTEO SIMPLE
# ============================================================================

@dataclass
class OperationCount:
    """Contador de operaciones en un bloque de código"""
    assignments: int = 0
    comparisons: int = 0
    arithmetic_ops: int = 0
    function_calls: int = 0
    array_accesses: int = 0


class CountingVisitor(ASTVisitor):
    """
    Visitor que cuenta operaciones básicas.
    
    Útil para:
    - Verificar que el visitor funciona correctamente
    - Obtener estadísticas del código
    - Base para análisis más complejos
    """
    
    def __init__(self):
        self.counts: Dict[str, OperationCount] = {}
        self.current_procedure: Optional[str] = None
    
    def visit_program(self, node: ProgramNode) -> Dict[str, OperationCount]:
        """Visita el programa y retorna conteos por procedimiento"""
        for proc in node.procedures:
            proc.accept(self)
        return self.counts
    
    def visit_procedure(self, node: ProcedureNode) -> None:
        """Visita un procedimiento y cuenta operaciones"""
        self.current_procedure = node.name
        self.counts[node.name] = OperationCount()
        
        # Visitar cuerpo
        node.body.accept(self)
    
    def visit_for(self, node: ForNode) -> None:
        """Cuenta operaciones en FOR"""
        # La comparación del FOR cuenta como 1
        self._increment('comparisons')
        
        # Visitar cuerpo
        node.body.accept(self)
    
    def visit_while(self, node: WhileNode) -> None:
        """Cuenta operaciones en WHILE"""
        # La condición cuenta como comparación
        self._increment('comparisons')
        
        # Visitar condición y cuerpo
        node.condition.accept(self)
        node.body.accept(self)
    
    def visit_repeat(self, node: RepeatNode) -> None:
        """Cuenta operaciones en REPEAT"""
        self._increment('comparisons')
        node.condition.accept(self)
        node.body.accept(self)
    
    def visit_if(self, node: IfNode) -> None:
        """Cuenta operaciones en IF"""
        self._increment('comparisons')
        
        node.condition.accept(self)
        node.then_block.accept(self)
        if node.else_block:
            node.else_block.accept(self)
    
    def visit_assignment(self, node: AssignmentNode) -> None:
        """Cuenta asignaciones"""
        self._increment('assignments')
        
        # Visitar subexpresiones
        node.target.accept(self)
        node.value.accept(self)
    
    def visit_binary_op(self, node: BinaryOpNode) -> None:
        """Cuenta operaciones aritméticas/comparaciones"""
        if node.op in ['+', '-', '*', '/', '^', 'mod', 'div']:
            self._increment('arithmetic_ops')
        elif node.op in ['<', '>', '≤', '≥', '=', '≠', 'and', 'or']:
            self._increment('comparisons')
        
        # Visitar operandos
        node.left.accept(self)
        node.right.accept(self)
    
    def visit_array_access(self, node: ArrayAccessNode) -> None:
        """Cuenta accesos a arrays"""
        self._increment('array_accesses')
        
        # Visitar índices
        for index in node.indices:
            if isinstance(index, RangeNode):
                index.accept(self)
            else:
                index.accept(self)
    
    def visit_function_call(self, node: FunctionCallNode) -> None:
        """Cuenta llamadas a funciones"""
        self._increment('function_calls')
        
        # Visitar argumentos
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_call_statement(self, node: CallStatementNode) -> None:
        """Cuenta llamadas a procedimientos"""
        self._increment('function_calls')
        
        for arg in node.arguments:
            arg.accept(self)
    
    def _increment(self, counter: str) -> None:
        """Helper para incrementar un contador"""
        if self.current_procedure:
            count = self.counts[self.current_procedure]
            setattr(count, counter, getattr(count, counter) + 1)


# ============================================================================
# EJEMPLO DE USO Y TESTS
# ============================================================================

def demo_counting_visitor():
    """Demuestra el uso del CountingVisitor"""
    from parser.parser import parse
    
    code = """
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
    """
    
    print("="*70)
    print("DEMO: CountingVisitor")
    print("="*70)
    print("\nCódigo:")
    print(code)
    
    # Parsear
    ast = parse(code)
    
    # Aplicar visitor
    visitor = CountingVisitor()
    counts = visitor.visit_program(ast)
    
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)
    
    for proc_name, count in counts.items():
        total_ops = (
            count.assignments +
            count.comparisons +
            count.arithmetic_ops +
            count.function_calls +
            count.array_accesses
        )

        print("\n" + "=" * 70)
        print("RESULTADOS")
        print("=" * 70)
        print(f"\nProcedimiento: {proc_name}")
        print(f"  Asignaciones:      {count.assignments}")
        print(f"  Comparaciones:     {count.comparisons}")
        print(f"  Operaciones arit.: {count.arithmetic_ops}")
        print(f"  Llamadas:          {count.function_calls}")
        print(f"  Accesos a arrays:  {count.array_accesses}")
        print(f"  TOTAL:             {total_ops}\n")



if __name__ == "__main__":
    demo_counting_visitor()