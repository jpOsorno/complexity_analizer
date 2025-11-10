"""
Transformer: Convierte árbol de Lark a AST personalizado
========================================================

Este módulo transforma el árbol genérico de Lark en nuestras
clases de nodos específicas definidas en nodes.py

El transformer usa el patrón Visitor de Lark, donde cada método
corresponde a una regla de la gramática.
"""

from lark import Transformer, Token, Tree
from typing import List, Union, Optional, Any

# Importar todos los nodos que definimos
from nodes import (
    # Programa
    ProgramNode, ClassDefNode, ProcedureNode,
    
    # Parámetros
    SimpleParamNode, ArrayParamNode, ClassParamNode,
    
    # Bloques y declaraciones
    BlockNode, ArrayDeclNode, ObjectDeclNode,
    
    # Sentencias
    ForNode, WhileNode, RepeatNode, IfNode,
    AssignmentNode, CallStatementNode, ReturnNode,
    
    # LValues
    VariableLValueNode, ArrayLValueNode, ObjectLValueNode,
    
    # Expresiones
    BinaryOpNode, UnaryOpNode,
    NumberNode, StringNode, BooleanNode, NullNode,
    IdentifierNode, ArrayAccessNode, RangeNode,
    ObjectAccessNode, FunctionCallNode,
    LengthNode, CeilingNode, FloorNode,
)


# ============================================================================
# TRANSFORMER PRINCIPAL
# ============================================================================

class PseudocodeTransformer(Transformer):
    """
    Transforma el árbol de Lark en nuestro AST personalizado.
    
    Cada método transforma una regla específica de la gramática.
    Los nombres de los métodos deben coincidir EXACTAMENTE con los
    nombres de las reglas en grammar.lark
    
    IMPORTANTE: Configuramos visit_tokens=False para controlar
    manualmente cuándo un Token se convierte en nodo.
    """
    
    # ========================================================================
    # HELPERS INTERNOS
    # ========================================================================
    
    @staticmethod
    def _extract_string(item) -> str:
        """
        Extrae un string de un Token, IdentifierNode, o string directo.
        
        Args:
            item: Token, IdentifierNode, o str
            
        Returns:
            String limpio
        """
        if isinstance(item, Token):
            return item.value
        elif isinstance(item, IdentifierNode):
            return item.name
        elif isinstance(item, str):
            return item
        else:
            return str(item)
    
    @staticmethod
    def _extract_number(item) -> Union[int, float]:
        """
        Extrae un número de un Token, NumberNode, o número directo.
        """
        if isinstance(item, Token):
            value = item.value
            return float(value) if '.' in value else int(value)
        elif isinstance(item, NumberNode):
            return item.value
        elif isinstance(item, (int, float)):
            return item
        else:
            return int(item)
    
    @staticmethod
    def _token_to_node(item):
        """
        Convierte un Token en el nodo apropiado.
        
        Args:
            item: Token, nodo, o valor primitivo
            
        Returns:
            Nodo apropiado (NumberNode, IdentifierNode, etc.)
        """
        if isinstance(item, Token):
            if item.type == 'NUMBER':
                value = item.value
                return NumberNode(value=float(value) if '.' in value else int(value))
            elif item.type == 'IDENTIFIER':
                return IdentifierNode(name=item.value)
            elif item.type == 'STRING':
                # Remover comillas
                return StringNode(value=item.value[1:-1])
            elif item.type == 'BOOLEAN':
                return BooleanNode(value=item.value in ('T', 'true'))
            elif item.type == 'NULL':
                return NullNode()
            else:
                # Otro tipo de token, retornar como está
                return item
        else:
            # Ya es un nodo o valor
            return item
    
    # ========================================================================
    # PROGRAMA Y DEFINICIONES
    # ========================================================================
    
    def program(self, items):
        """
        program: (class_definition)* procedure_definition+
        
        Separa clases de procedimientos
        """
        classes = []
        procedures = []
        
        for item in items:
            if isinstance(item, ClassDefNode):
                classes.append(item)
            elif isinstance(item, ProcedureNode):
                procedures.append(item)
        
        return ProgramNode(classes=classes, procedures=procedures)
    
    def class_definition(self, items):
        """
        class_definition: IDENTIFIER "{" attribute_list "}"
        """
        class_name = str(items[0])  # IDENTIFIER
        attributes = items[1] if len(items) > 1 else []
        
        return ClassDefNode(name=class_name, attributes=attributes)
    
    def attribute_list(self, items):
        """
        attribute_list: IDENTIFIER+
        """
        return [str(item) for item in items]
    
    def procedure_definition(self, items):
        """
        procedure_definition: IDENTIFIER "(" parameter_list? ")" block
        """
        # Usar helper para extraer el nombre
        name = self._extract_string(items[0])
        
        # Puede tener parámetros o no
        if len(items) == 2:
            # Sin parámetros: (nombre, block)
            parameters = []
            body = items[1]
        else:
            # Con parámetros: (nombre, params, block)
            parameters = items[1] if isinstance(items[1], list) else []
            body = items[2]
        
        return ProcedureNode(name=name, parameters=parameters, body=body)
    
    def parameter_list(self, items):
        """
        parameter_list: parameter ("," parameter)*
        """
        return list(items)
    
    def parameter(self, items):
        """
        parameter: class_param | array_param | simple_param
        
        Este método delega a los submétodos, pero asegura que 
        retornamos el nodo correcto
        """
        # Si items tiene un solo elemento y ya es un nodo de parámetro, retornarlo
        if len(items) == 1:
            return items[0]
        
        # Si hay múltiples items, procesarlos
        # (esto no debería pasar con nuestra gramática, pero por si acaso)
        return items[0]
    
    def simple_param(self, items):
        """
        simple_param: IDENTIFIER
        """
        name = self._extract_string(items[0])
        return SimpleParamNode(name=name)
    
    def array_param(self, items):
        """
        array_param: IDENTIFIER ("[" NUMBER? "]")+
        """
        name = self._extract_string(items[0])
        dimensions = []
        
        # Los items restantes son los números dentro de []
        for item in items[1:]:
            if item is None:
                dimensions.append(None)  # [] sin tamaño
            else:
                dimensions.append(self._extract_number(item))
        
        return ArrayParamNode(name=name, dimensions=dimensions)
    
    def class_param(self, items):
        """
        class_param: "Clase" IDENTIFIER
        """
        # El identificador puede estar en items[0] o items[1]
        if len(items) > 1:
            name = self._extract_string(items[1])
        else:
            name = self._extract_string(items[0])
        
        return ClassParamNode(name=name, class_name="Clase")
    
    # ========================================================================
    # BLOQUES Y DECLARACIONES
    # ========================================================================
    
    def block(self, items):
        """
        block: "begin" declaration* statement* "end"
        """
        declarations = []
        statements = []
        
        for item in items:
            if isinstance(item, (ArrayDeclNode, ObjectDeclNode)):
                declarations.append(item)
            elif item is not None:  # Ignorar None de tokens
                statements.append(item)
        
        return BlockNode(declarations=declarations, statements=statements)
    
    def local_array_decl(self, items):
        """
        local_array_decl: IDENTIFIER ("[" expression? "]")+
        """
        name = str(items[0])
        dimensions = []
        
        for item in items[1:]:
            dimensions.append(item)  # None si no hay expresión
        
        return ArrayDeclNode(name=name, dimensions=dimensions)
    
    def local_object_decl(self, items):
        """
        local_object_decl: "Clase" IDENTIFIER
        """
        class_name = "Clase"
        object_name = str(items[1]) if len(items) > 1 else str(items[0])
        
        return ObjectDeclNode(class_name=class_name, object_name=object_name)
    
    # ========================================================================
    # SENTENCIAS
    # ========================================================================
    
    def for_statement(self, items):
        """
        for_statement: "for" IDENTIFIER "←" expression "to" expression "do" block
        """
        variable = self._extract_string(items[0])
        start = self._token_to_node(items[1])
        end = self._token_to_node(items[2])
        body = items[3]
        
        return ForNode(variable=variable, start=start, end=end, body=body)
    
    def while_statement(self, items):
        """
        while_statement: "while" "(" boolean_expression ")" "do" block
        """
        condition = self._token_to_node(items[0])
        body = items[1]
        
        return WhileNode(condition=condition, body=body)
    
    def repeat_statement(self, items):
        """
        repeat_statement: "repeat" statement+ "until" "(" boolean_expression ")"
        """
        # Todos los items excepto el último son statements
        # El último es la condición
        statements = items[:-1]
        condition = self._token_to_node(items[-1])
        
        # Crear un bloque con los statements
        body = BlockNode(statements=list(statements))
        
        return RepeatNode(body=body, condition=condition)
    
    def if_statement(self, items):
        """
        if_statement: "if" "(" boolean_expression ")" "then" block ("else" block)?
        """
        condition = self._token_to_node(items[0])
        then_block = items[1]
        else_block = items[2] if len(items) > 2 else None
        
        return IfNode(condition=condition, then_block=then_block, else_block=else_block)
    
    def assignment(self, items):
        """
        assignment: lvalue "←" expression
        """
        target = items[0]  # Ya está transformado por lvalue()
        value = self._token_to_node(items[1])
        
        return AssignmentNode(target=target, value=value)
    
    def call_statement(self, items):
        """
        call_statement: "call" IDENTIFIER "(" argument_list? ")"
        """
        name = str(items[0])
        arguments = items[1] if len(items) > 1 and items[1] is not None else []
        
        return CallStatementNode(name=name, arguments=arguments)
    
    def return_statement(self, items):
        """
        return_statement: "return" expression?
        """
        value = self._token_to_node(items[0]) if items else None
        
        return ReturnNode(value=value)
    
    def argument_list(self, items):
        """
        argument_list: expression ("," expression)*
        """
        return list(items)
    
    # ========================================================================
    # LVALUES
    # ========================================================================
    
    def lvalue(self, items):
        """
        lvalue puede ser:
        - IDENTIFIER (variable simple)
        - IDENTIFIER ("[" expression "]")+ (array)
        - IDENTIFIER ("." IDENTIFIER)+ (objeto)
        - IDENTIFIER ("[" expression "]")+ ("." IDENTIFIER)+ (combinado)
        
        La gramática inline define estas estructuras directamente.
        """
        # Transformar todos los items primero
        transformed_items = [self._token_to_node(item) for item in items]
        
        # Extraer el nombre base
        name = self._extract_string(transformed_items[0])
        
        # Separar los items restantes en indices y campos
        indices = []
        fields = []
        
        i = 1
        while i < len(transformed_items):
            item = transformed_items[i]
            
            # Si es un nodo de expresión, es un índice de array
            if isinstance(item, (BinaryOpNode, NumberNode, IdentifierNode, 
                                UnaryOpNode, ArrayAccessNode, FunctionCallNode,
                                StringNode, BooleanNode)):
                indices.append(item)
            # Si es un string simple, es un campo de objeto
            elif isinstance(item, str):
                fields.append(item)
            # Si todavía es un Token IDENTIFIER (no transformado), es un campo
            elif isinstance(item, Token) and item.type == 'IDENTIFIER':
                fields.append(item.value)
            
            i += 1
        
        # Decidir qué tipo de lvalue crear basado en lo que encontramos
        if len(indices) > 0 and len(fields) == 0:
            # Solo índices: es un array
            return ArrayLValueNode(name=name, indices=indices)
        elif len(fields) > 0 and len(indices) == 0:
            # Solo campos: es un objeto
            return ObjectLValueNode(object_name=name, fields=fields)
        elif len(indices) > 0 and len(fields) > 0:
            # Ambos: tratamos como array por ahora
            # (casos como A[i].campo son raros)
            return ArrayLValueNode(name=name, indices=indices)
        else:
            # Ni índices ni campos: variable simple
            return VariableLValueNode(name=name)
    
    # ========================================================================
    # EXPRESIONES: OPERADORES BINARIOS CON ALIAS
    # ========================================================================
    
    # Operadores aritméticos
    def add(self, items):
        """Suma: left + right"""
        return BinaryOpNode(op="+", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def sub(self, items):
        """Resta: left - right"""
        return BinaryOpNode(op="-", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def mul(self, items):
        """Multiplicación: left * right"""
        return BinaryOpNode(op="*", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def div(self, items):
        """División real: left / right"""
        return BinaryOpNode(op="/", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def mod(self, items):
        """Módulo: left mod right"""
        return BinaryOpNode(op="mod", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def int_div(self, items):
        """División entera: left div right"""
        return BinaryOpNode(op="div", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def pow(self, items):
        """Potencia: base ^ exp"""
        return BinaryOpNode(op="^", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    # Operadores unarios
    def pos(self, items):
        """Unario positivo: +x"""
        return UnaryOpNode(op="+", operand=self._token_to_node(items[0]))
    
    def neg(self, items):
        """Unario negativo: -x"""
        return UnaryOpNode(op="-", operand=self._token_to_node(items[0]))
    
    # Operadores relacionales
    def lt(self, items):
        """Menor que: left < right"""
        return BinaryOpNode(op="<", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def gt(self, items):
        """Mayor que: left > right"""
        return BinaryOpNode(op=">", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def le(self, items):
        """Menor o igual: left ≤ right"""
        return BinaryOpNode(op="≤", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def ge(self, items):
        """Mayor o igual: left ≥ right"""
        return BinaryOpNode(op="≥", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def eq(self, items):
        """Igual: left = right"""
        return BinaryOpNode(op="=", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def ne(self, items):
        """Diferente: left ≠ right"""
        return BinaryOpNode(op="≠", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    # Operadores booleanos
    def and_op(self, items):
        """AND lógico: left and right"""
        return BinaryOpNode(op="and", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def or_op(self, items):
        """OR lógico: left or right"""
        return BinaryOpNode(op="or", 
                          left=self._token_to_node(items[0]), 
                          right=self._token_to_node(items[1]))
    
    def not_op(self, items):
        """NOT lógico: not operand"""
        return UnaryOpNode(op="not", operand=self._token_to_node(items[0]))
    
    # ========================================================================
    # EXPRESIONES: PRIMARIOS
    # ========================================================================
    
    def primary(self, items):
        """
        Primary puede ser muchas cosas diferentes.
        Lark ya resolvió la ambigüedad, solo retornamos el resultado.
        
        Puede ser:
        - NUMBER, STRING, BOOLEAN, NULL
        - IDENTIFIER (variable simple)
        - IDENTIFIER "[" ... "]" (array access)
        - IDENTIFIER "." ... (object access)
        - function_call, length, ceil, floor
        - "(" expression ")"
        """
        if len(items) == 1:
            # Transformar el token si es necesario
            return self._token_to_node(items[0])
        
        # Si hay múltiples items, necesitamos construir el nodo apropiado
        # Transformar todos los items primero
        transformed_items = [self._token_to_node(item) for item in items]
        
        # Primer item es el nombre
        if isinstance(transformed_items[0], Token) and transformed_items[0].type == 'IDENTIFIER':
            name = transformed_items[0].value
        elif isinstance(transformed_items[0], IdentifierNode):
            name = transformed_items[0].name
        elif isinstance(transformed_items[0], str):
            name = transformed_items[0]
        else:
            # Ya es un nodo completo
            return transformed_items[0]
        
        # Detectar si es array access o object access
        has_indices = False
        has_fields = False
        indices = []
        fields = []
        
        for item in transformed_items[1:]:
            # Si es un nodo de expresión, es índice de array
            if isinstance(item, (BinaryOpNode, NumberNode, IdentifierNode, 
                                UnaryOpNode, ArrayAccessNode, FunctionCallNode,
                                RangeNode, StringNode, BooleanNode)):
                indices.append(item)
                has_indices = True
            # Si es string, es campo de objeto
            elif isinstance(item, str):
                fields.append(item)
                has_fields = True
            # Si es Token IDENTIFIER (no transformado), es campo
            elif isinstance(item, Token) and item.type == 'IDENTIFIER':
                fields.append(item.value)
                has_fields = True
        
        if has_indices:
            return ArrayAccessNode(name=name, indices=indices)
        elif has_fields:
            return ObjectAccessNode(object_name=name, fields=fields)
        else:
            return IdentifierNode(name=name)
    
    # ========================================================================
    # EXPRESIONES: LITERALES Y VALORES
    # ========================================================================
    
    # Los tokens NUMBER, STRING, BOOLEAN, IDENTIFIER, NULL
    # se transforman automáticamente por _token_to_node()
    # cuando se procesan en primary, lvalue, etc.
    
    # No necesitamos métodos separados para ellos porque
    # visit_tokens=False evita que se llamen automáticamente.
    
    def array_index(self, items):
        """
        array_index: expression | expression ".." expression
        """
        if len(items) == 1:
            return items[0]
        else:
            # Es un rango
            return RangeNode(start=items[0], end=items[1])
    
    def function_call(self, items):
        """
        function_call: "call" IDENTIFIER "(" argument_list? ")"
        """
        name = self._extract_string(items[0])
        arguments = items[1] if len(items) > 1 and items[1] is not None else []
        
        return FunctionCallNode(name=name, arguments=arguments)
    
    def length_function(self, items):
        """
        length_function: "length" "(" IDENTIFIER ")"
        """
        array_name = self._extract_string(items[0])
        return LengthNode(array_name=array_name)
    
    def ceiling_function(self, items):
        """
        ceiling_function: "┌" expression "┐" | "ceil" "(" expression ")"
        """
        return CeilingNode(expression=self._token_to_node(items[0]))
    
    def floor_function(self, items):
        """
        floor_function: "└" expression "┘" | "floor" "(" expression ")"
        """
        return FloorNode(expression=self._token_to_node(items[0]))
    
    def ceiling_function(self, items):
        """
        ceiling_function: "┌" expression "┐" | "ceil" "(" expression ")"
        """
        return CeilingNode(expression=items[0])
    
    def floor_function(self, items):
        """
        floor_function: "└" expression "┘" | "floor" "(" expression ")"
        """
        return FloorNode(expression=items[0])
    
    # ========================================================================
    # CASOS ESPECIALES
    # ========================================================================
    
    # NULL se maneja en _token_to_node()
    # Ya no necesitamos un método separado


# ============================================================================
# FUNCIÓN HELPER PARA USAR EL TRANSFORMER
# ============================================================================

def transform_to_ast(lark_tree: Tree) -> ProgramNode:
    """
    Transforma un árbol de Lark en nuestro AST personalizado.
    
    Args:
        lark_tree: Árbol generado por el parser de Lark
        
    Returns:
        ProgramNode: Raíz del AST personalizado
        
    Example:
        >>> from lark import Lark
        >>> parser = Lark.open('grammar.lark', parser='earley', start='program')
        >>> tree = parser.parse(pseudocode)
        >>> ast = transform_to_ast(tree)
        >>> print(ast.procedures[0].name)
    """
    # visit_tokens=False hace que NO se llamen automáticamente 
    # los métodos para tokens en todos los contextos
    # Solo se transforman cuando están en expresiones
    transformer = PseudocodeTransformer(visit_tokens=False)
    return transformer.transform(lark_tree)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    from lark import Lark
    import os
    
    # Cargar gramática
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grammar_path = os.path.join(script_dir, '../parser', 'grammar.lark')
    
    with open(grammar_path, 'r', encoding='utf-8') as f:
        grammar = f.read()
    
    parser = Lark(grammar, parser='earley', start='program')
    
    # Código de prueba
    test_code = """
Simple()
begin
    x ← 5
end
    """
    
    print("="*70)
    print("TRANSFORMACIÓN LARK → AST")
    print("="*70)
    
    # Parse
    lark_tree = parser.parse(test_code)
    print("\n1. Árbol de Lark:")
    print(lark_tree.pretty())
    
    # Transform
    ast = transform_to_ast(lark_tree)
    print("\n2. AST Personalizado:")
    print(ast)
    print(f"\nProcedimiento: {ast.procedures[0].name}")
    print(f"Parámetros: {[p.name for p in ast.procedures[0].parameters]}")
    print(f"Tipo de body: {type(ast.procedures[0].body).__name__}")