"""
Nodos del Árbol de Sintaxis Abstracta (AST)
===========================================

Este módulo define todas las clases de nodos que representan
la estructura de un algoritmo en pseudocódigo.

Cada nodo tiene:
- __init__: Constructor con los datos del nodo
- __repr__: Representación legible para debugging
- accept: Método para el patrón Visitor (análisis)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from abc import ABC, abstractmethod


# ============================================================================
# CLASE BASE: Todos los nodos heredan de esta
# ============================================================================

class ASTNode(ABC):
    """
    Clase base abstracta para todos los nodos del AST.
    
    Usa el patrón Visitor para permitir diferentes tipos de
    análisis sin modificar las clases de nodos.
    """
    
    @abstractmethod
    def accept(self, visitor):
        """
        Acepta un visitor que procesará este nodo.
        
        Args:
            visitor: Objeto que implementa métodos visit_* para cada tipo de nodo
            
        Returns:
            El resultado del procesamiento del visitor
        """
        pass
    
    def __repr__(self):
        """Representación por defecto basada en atributos"""
        attrs = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({attrs})'


# ============================================================================
# NODO RAÍZ: Programa completo
# ============================================================================

@dataclass
class ProgramNode(ASTNode):
    """
    Representa el programa completo.
    
    Attributes:
        classes: Lista de definiciones de clases
        procedures: Lista de procedimientos/funciones
    """
    classes: List['ClassDefNode'] = field(default_factory=list)
    procedures: List['ProcedureNode'] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_program(self)


# ============================================================================
# DEFINICIONES: Clases y Procedimientos
# ============================================================================

@dataclass
class ClassDefNode(ASTNode):
    """
    Definición de una clase.
    
    Ejemplo: Persona {nombre edad direccion}
    
    Attributes:
        name: Nombre de la clase
        attributes: Lista de nombres de atributos
    """
    name: str
    attributes: List[str] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_class_def(self)


@dataclass
class ProcedureNode(ASTNode):
    """
    Definición de un procedimiento o función.
    
    Ejemplo: BubbleSort(A[], n)
    
    Attributes:
        name: Nombre del procedimiento
        parameters: Lista de parámetros
        body: Bloque de código del procedimiento
    """
    name: str
    parameters: List['ParameterNode']
    body: 'BlockNode'
    
    def accept(self, visitor):
        return visitor.visit_procedure(self)


# ============================================================================
# PARÁMETROS: Diferentes tipos de parámetros
# ============================================================================

@dataclass
class ParameterNode(ASTNode):
    """Clase base para parámetros"""
    name: str
    
    @abstractmethod
    def accept(self, visitor):
        pass


@dataclass
class SimpleParamNode(ParameterNode):
    """
    Parámetro simple: nombre
    
    Ejemplo: n, x, valor
    """
    
    def accept(self, visitor):
        return visitor.visit_simple_param(self)


@dataclass
class ArrayParamNode(ParameterNode):
    """
    Parámetro array: nombre[dim1][dim2]...
    
    Ejemplo: A[], matriz[n][m]
    
    Attributes:
        name: Nombre del array
        dimensions: Lista de dimensiones (puede ser vacía para [])
    """
    dimensions: List[Optional[int]] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_array_param(self)


@dataclass
class ClassParamNode(ParameterNode):
    """
    Parámetro de tipo clase: Clase nombreObjeto
    
    Ejemplo: Clase persona, Clase casa
    
    Attributes:
        name: Nombre del objeto
        class_name: Nombre de la clase (extraído de "Clase")
    """
    class_name: str = "Clase"
    
    def accept(self, visitor):
        return visitor.visit_class_param(self)


# ============================================================================
# BLOQUES Y DECLARACIONES
# ============================================================================

@dataclass
class BlockNode(ASTNode):
    """
    Bloque de código: begin ... end
    
    Attributes:
        declarations: Declaraciones locales (arrays, objetos)
        statements: Lista de sentencias
    """
    declarations: List[Union['ArrayDeclNode', 'ObjectDeclNode']] = field(default_factory=list)
    statements: List['StatementNode'] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_block(self)


@dataclass
class ArrayDeclNode(ASTNode):
    """
    Declaración de array local: A[n] o matriz[n][m]
    
    Attributes:
        name: Nombre del array
        dimensions: Lista de expresiones para los tamaños
    """
    name: str
    dimensions: List[Optional['ExpressionNode']] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_array_decl(self)


@dataclass
class ObjectDeclNode(ASTNode):
    """
    Declaración de objeto: Clase nombreObjeto
    
    Attributes:
        class_name: Nombre de la clase
        object_name: Nombre del objeto
    """
    class_name: str
    object_name: str
    
    def accept(self, visitor):
        return visitor.visit_object_decl(self)


# ============================================================================
# SENTENCIAS (STATEMENTS)
# ============================================================================

class StatementNode(ASTNode):
    """Clase base para todas las sentencias"""
    pass


@dataclass
class ForNode(StatementNode):
    """
    Ciclo FOR: for variable ← start to end do begin ... end
    
    Attributes:
        variable: Nombre de la variable de control
        start: Expresión inicial
        end: Expresión final
        body: Bloque de código del ciclo
    """
    variable: str
    start: 'ExpressionNode'
    end: 'ExpressionNode'
    body: BlockNode
    
    def accept(self, visitor):
        return visitor.visit_for(self)


@dataclass
class WhileNode(StatementNode):
    """
    Ciclo WHILE: while (condition) do begin ... end
    
    Attributes:
        condition: Expresión booleana
        body: Bloque de código del ciclo
    """
    condition: 'ExpressionNode'
    body: BlockNode
    
    def accept(self, visitor):
        return visitor.visit_while(self)


@dataclass
class RepeatNode(StatementNode):
    """
    Ciclo REPEAT: repeat ... until (condition)
    
    Attributes:
        body: Bloque de código del ciclo
        condition: Expresión booleana (se evalúa al final)
    """
    body: BlockNode
    condition: 'ExpressionNode'
    
    def accept(self, visitor):
        return visitor.visit_repeat(self)


@dataclass
class IfNode(StatementNode):
    """
    Condicional IF: if (condition) then begin ... end else begin ... end
    
    Attributes:
        condition: Expresión booleana
        then_block: Bloque si la condición es verdadera
        else_block: Bloque si la condición es falsa (opcional)
    """
    condition: 'ExpressionNode'
    then_block: BlockNode
    else_block: Optional[BlockNode] = None
    
    def accept(self, visitor):
        return visitor.visit_if(self)


@dataclass
class AssignmentNode(StatementNode):
    """
    Asignación: target ← value
    
    Attributes:
        target: LValue (variable, array, objeto)
        value: Expresión a asignar
    """
    target: 'LValueNode'
    value: 'ExpressionNode'
    
    def accept(self, visitor):
        return visitor.visit_assignment(self)


@dataclass
class CallStatementNode(StatementNode):
    """
    Llamada a procedimiento como sentencia: call Procedimiento(args)
    
    Attributes:
        name: Nombre del procedimiento
        arguments: Lista de expresiones (argumentos)
    """
    name: str
    arguments: List['ExpressionNode'] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_call_statement(self)


@dataclass
class ReturnNode(StatementNode):
    """
    Retorno de función: return expression
    
    Attributes:
        value: Expresión a retornar (None si es return sin valor)
    """
    value: Optional['ExpressionNode'] = None
    
    def accept(self, visitor):
        return visitor.visit_return(self)


# ============================================================================
# LVALUES: Lado izquierdo de asignaciones
# ============================================================================

class LValueNode(ASTNode):
    """Clase base para lvalues (targets de asignación)"""
    pass


@dataclass
class VariableLValueNode(LValueNode):
    """
    Variable simple: x
    
    Attributes:
        name: Nombre de la variable
    """
    name: str
    
    def accept(self, visitor):
        return visitor.visit_variable_lvalue(self)


@dataclass
class ArrayLValueNode(LValueNode):
    """
    Acceso a array: A[i], matriz[i][j]
    
    Attributes:
        name: Nombre del array
        indices: Lista de expresiones para los índices
    """
    name: str
    indices: List['ExpressionNode']
    
    def accept(self, visitor):
        return visitor.visit_array_lvalue(self)


@dataclass
class ObjectLValueNode(LValueNode):
    """
    Acceso a campo de objeto: objeto.campo, objeto.campo.subcampo
    
    Attributes:
        object_name: Nombre del objeto base
        fields: Lista de nombres de campos
    """
    object_name: str
    fields: List[str]
    
    def accept(self, visitor):
        return visitor.visit_object_lvalue(self)


# ============================================================================
# EXPRESIONES
# ============================================================================

class ExpressionNode(ASTNode):
    """Clase base para todas las expresiones"""
    pass


@dataclass
class BinaryOpNode(ExpressionNode):
    """
    Operación binaria: left op right
    
    Operadores: +, -, *, /, mod, div, ^, <, >, ≤, ≥, =, ≠, and, or
    
    Attributes:
        op: Operador como string
        left: Expresión izquierda
        right: Expresión derecha
    """
    op: str
    left: ExpressionNode
    right: ExpressionNode
    
    def accept(self, visitor):
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOpNode(ExpressionNode):
    """
    Operación unaria: op operand
    
    Operadores: -, +, not
    
    Attributes:
        op: Operador como string
        operand: Expresión
    """
    op: str
    operand: ExpressionNode
    
    def accept(self, visitor):
        return visitor.visit_unary_op(self)


@dataclass
class NumberNode(ExpressionNode):
    """
    Literal numérico: 42, 3.14
    
    Attributes:
        value: Valor numérico (int o float)
    """
    value: Union[int, float]
    
    def accept(self, visitor):
        return visitor.visit_number(self)


@dataclass
class StringNode(ExpressionNode):
    """
    Literal string: "hola", 'mundo'
    
    Attributes:
        value: Contenido del string (sin comillas)
    """
    value: str
    
    def accept(self, visitor):
        return visitor.visit_string(self)


@dataclass
class BooleanNode(ExpressionNode):
    """
    Literal booleano: T, F, true, false
    
    Attributes:
        value: True o False
    """
    value: bool
    
    def accept(self, visitor):
        return visitor.visit_boolean(self)


@dataclass
class NullNode(ExpressionNode):
    """
    Literal NULL
    """
    
    def accept(self, visitor):
        return visitor.visit_null(self)


@dataclass
class IdentifierNode(ExpressionNode):
    """
    Identificador (variable): x, contador, resultado
    
    Attributes:
        name: Nombre de la variable
    """
    name: str
    
    def accept(self, visitor):
        return visitor.visit_identifier(self)


@dataclass
class ArrayAccessNode(ExpressionNode):
    """
    Acceso a array: A[i], matriz[i][j], A[1..n]
    
    Attributes:
        name: Nombre del array
        indices: Lista de índices (pueden ser rangos)
    """
    name: str
    indices: List[Union['ExpressionNode', 'RangeNode']]
    
    def accept(self, visitor):
        return visitor.visit_array_access(self)


@dataclass
class RangeNode(ExpressionNode):
    """
    Rango de array: 1..n
    
    Attributes:
        start: Expresión inicial
        end: Expresión final
    """
    start: ExpressionNode
    end: ExpressionNode
    
    def accept(self, visitor):
        return visitor.visit_range(self)


@dataclass
class ObjectAccessNode(ExpressionNode):
    """
    Acceso a campo de objeto: objeto.campo
    
    Attributes:
        object_name: Nombre del objeto base
        fields: Lista de nombres de campos (para acceso anidado)
    """
    object_name: str
    fields: List[str]
    
    def accept(self, visitor):
        return visitor.visit_object_access(self)


@dataclass
class FunctionCallNode(ExpressionNode):
    """
    Llamada a función en expresión: call Factorial(n-1)
    
    Attributes:
        name: Nombre de la función
        arguments: Lista de expresiones (argumentos)
    """
    name: str
    arguments: List[ExpressionNode] = field(default_factory=list)
    
    def accept(self, visitor):
        return visitor.visit_function_call(self)


@dataclass
class LengthNode(ExpressionNode):
    """
    Función length: length(A)
    
    Attributes:
        array_name: Nombre del array
    """
    array_name: str
    
    def accept(self, visitor):
        return visitor.visit_length(self)


@dataclass
class CeilingNode(ExpressionNode):
    """
    Función ceiling: ┌x┐ o ceil(x)
    
    Attributes:
        expression: Expresión a redondear hacia arriba
    """
    expression: ExpressionNode
    
    def accept(self, visitor):
        return visitor.visit_ceiling(self)


@dataclass
class FloorNode(ExpressionNode):
    """
    Función floor: └x┘ o floor(x)
    
    Attributes:
        expression: Expresión a redondear hacia abajo
    """
    expression: ExpressionNode
    
    def accept(self, visitor):
        return visitor.visit_floor(self)


# ============================================================================
# UTILIDADES: Función helper para crear el AST fácilmente
# ============================================================================

def create_simple_assignment(var_name: str, value: Union[int, str]) -> AssignmentNode:
    """
    Helper para crear asignaciones simples: x ← 5
    
    Args:
        var_name: Nombre de la variable
        value: Valor a asignar (número o string)
        
    Returns:
        AssignmentNode configurado
    """
    target = VariableLValueNode(var_name)
    
    if isinstance(value, int):
        expr = NumberNode(value)
    elif isinstance(value, str):
        expr = StringNode(value)
    else:
        raise ValueError(f"Tipo de valor no soportado: {type(value)}")
    
    return AssignmentNode(target, expr)


def create_for_loop(variable: str, start: int, end: Union[int, str], 
                     statements: List[StatementNode]) -> ForNode:
    """
    Helper para crear ciclos FOR simples
    
    Args:
        variable: Nombre de la variable de control
        start: Valor inicial (entero)
        end: Valor final (entero o identificador)
        statements: Lista de sentencias del cuerpo
        
    Returns:
        ForNode configurado
    """
    start_expr = NumberNode(start)
    
    if isinstance(end, int):
        end_expr = NumberNode(end)
    elif isinstance(end, str):
        end_expr = IdentifierNode(end)
    else:
        raise ValueError(f"Tipo de end no soportado: {type(end)}")
    
    body = BlockNode(statements=statements)
    
    return ForNode(variable, start_expr, end_expr, body)