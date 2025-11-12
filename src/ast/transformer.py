"""
Transformer: Convierte árbol de Lark a AST personalizado (VERSIÓN CORREGIDA)
============================================================================

Correcciones principales:
1. Transformación consistente de Tokens a nodos
2. Manejo correcto de lvalues complejos
3. Soporte completo para acceso a objetos
"""

from lark import Transformer, Token, Tree
from typing import List, Union, Optional, Any

from .nodes import (
    ProgramNode, ClassDefNode, ProcedureNode,
    SimpleParamNode, ArrayParamNode, ClassParamNode,
    BlockNode, ArrayDeclNode, ObjectDeclNode,
    ForNode, WhileNode, RepeatNode, IfNode,
    AssignmentNode, CallStatementNode, ReturnNode,
    VariableLValueNode, ArrayLValueNode, ObjectLValueNode,
    BinaryOpNode, UnaryOpNode,
    NumberNode, StringNode, BooleanNode, NullNode,
    IdentifierNode, ArrayAccessNode, RangeNode,
    ObjectAccessNode, FunctionCallNode,
    LengthNode, CeilingNode, FloorNode,
)


class PseudocodeTransformer(Transformer):
    """Transforma árbol de Lark a AST personalizado"""
    
    # ========================================================================
    # HELPERS MEJORADOS
    # ========================================================================
    
    @staticmethod
    def _to_node(item):
        """
        Convierte CUALQUIER item a un nodo apropiado.
        
        Esta es la función clave que previene errores de Token.
        """
        # Ya es un nodo del AST
        if isinstance(item, (
            ProgramNode, ProcedureNode, BlockNode,
            ForNode, WhileNode, RepeatNode, IfNode,
            AssignmentNode, CallStatementNode, ReturnNode,
            VariableLValueNode, ArrayLValueNode, ObjectLValueNode,
            BinaryOpNode, UnaryOpNode,
            NumberNode, StringNode, BooleanNode, NullNode,
            IdentifierNode, ArrayAccessNode, RangeNode,
            ObjectAccessNode, FunctionCallNode,
            LengthNode, CeilingNode, FloorNode,
            SimpleParamNode, ArrayParamNode, ClassParamNode,
            ArrayDeclNode, ObjectDeclNode, ClassDefNode
        )):
            return item
        
        # Es un Token - transformar según tipo
        if isinstance(item, Token):
            if item.type == 'NUMBER':
                value = item.value
                return NumberNode(value=float(value) if '.' in value else int(value))
            
            elif item.type == 'IDENTIFIER':
                return IdentifierNode(name=item.value)
            
            elif item.type == 'STRING':
                # Remover comillas
                content = item.value[1:-1] if len(item.value) >= 2 else item.value
                return StringNode(value=content)
            
            elif item.type == 'BOOLEAN':
                return BooleanNode(value=item.value in ('T', 'true'))
            
            elif item.type == 'NULL':
                return NullNode()
            
            else:
                # Token desconocido - retornar como IdentifierNode
                return IdentifierNode(name=str(item.value))
        
        # Es un valor primitivo
        if isinstance(item, (int, float)):
            return NumberNode(value=item)
        
        if isinstance(item, str):
            return IdentifierNode(name=item)
        
        if isinstance(item, bool):
            return BooleanNode(value=item)
        
        # Lista - transformar cada elemento
        if isinstance(item, list):
            return [PseudocodeTransformer._to_node(x) for x in item]
        
        # None
        if item is None:
            return None
        
        # Caso por defecto
        return item
    
    @staticmethod
    def _extract_name(item) -> str:
        """Extrae un nombre de string de cualquier item"""
        if isinstance(item, Token):
            return item.value
        elif isinstance(item, IdentifierNode):
            return item.name
        elif isinstance(item, str):
            return item
        else:
            return str(item)
    
    # ========================================================================
    # PROGRAMA Y DEFINICIONES
    # ========================================================================
    
    def program(self, items):
        """program: (class_definition)* procedure_definition+"""
        classes = []
        procedures = []
        
        for item in items:
            if isinstance(item, ClassDefNode):
                classes.append(item)
            elif isinstance(item, ProcedureNode):
                procedures.append(item)
        
        return ProgramNode(classes=classes, procedures=procedures)
    
    def class_definition(self, items):
        """class_definition: IDENTIFIER "{" attribute_list "}"  """
        class_name = self._extract_name(items[0])
        attributes = items[1] if len(items) > 1 else []
        return ClassDefNode(name=class_name, attributes=attributes)
    
    def attribute_list(self, items):
        """attribute_list: IDENTIFIER+"""
        return [self._extract_name(item) for item in items]
    
    def procedure_definition(self, items):
        """procedure_definition: IDENTIFIER "(" parameter_list? ")" block"""
        name = self._extract_name(items[0])
        
        if len(items) == 2:
            parameters = []
            body = items[1]
        else:
            parameters = items[1] if isinstance(items[1], list) else []
            body = items[2]
        
        return ProcedureNode(name=name, parameters=parameters, body=body)
    
    def parameter_list(self, items):
        """parameter_list: parameter ("," parameter)*"""
        return list(items)
    
    def parameter(self, items):
        """parameter: class_param | array_param | simple_param"""
        return items[0]
    
    def simple_param(self, items):
        """simple_param: IDENTIFIER"""
        name = self._extract_name(items[0])
        return SimpleParamNode(name=name)
    
    def array_param(self, items):
        """array_param: IDENTIFIER ("[" NUMBER? "]")+"""
        name = self._extract_name(items[0])
        dimensions = []
        
        for item in items[1:]:
            if item is None:
                dimensions.append(None)
            else:
                # Convertir a número
                if isinstance(item, Token):
                    val = item.value
                    dimensions.append(int(val) if '.' not in val else float(val))
                elif isinstance(item, NumberNode):
                    dimensions.append(item.value)
                else:
                    dimensions.append(item)
        
        return ArrayParamNode(name=name, dimensions=dimensions)
    
    def class_param(self, items):
        """class_param: "Clase" IDENTIFIER"""
        name = self._extract_name(items[-1])  # Último item es el nombre
        return ClassParamNode(name=name, class_name="Clase")
    
    # ========================================================================
    # BLOQUES Y DECLARACIONES
    # ========================================================================
    
    def block(self, items):
        """block: "begin" declaration* statement* "end" """
        declarations = []
        statements = []
        
        for item in items:
            if isinstance(item, (ArrayDeclNode, ObjectDeclNode)):
                declarations.append(item)
            elif item is not None and not isinstance(item, Token):
                statements.append(item)
        
        return BlockNode(declarations=declarations, statements=statements)
    
    def local_array_decl(self, items):
        """local_array_decl: IDENTIFIER ("[" expression? "]")+"""
        name = self._extract_name(items[0])
        dimensions = []
        
        for item in items[1:]:
            dimensions.append(self._to_node(item))
        
        return ArrayDeclNode(name=name, dimensions=dimensions)
    
    def local_object_decl(self, items):
        """local_object_decl: "Clase" IDENTIFIER"""
        object_name = self._extract_name(items[-1])
        return ObjectDeclNode(class_name="Clase", object_name=object_name)
    
    # ========================================================================
    # SENTENCIAS
    # ========================================================================
    
    def for_statement(self, items):
        """for_statement: "for" IDENTIFIER "←" expression "to" expression "do" block"""
        variable = self._extract_name(items[0])
        start = self._to_node(items[1])
        end = self._to_node(items[2])
        body = items[3]
        
        return ForNode(variable=variable, start=start, end=end, body=body)
    
    def while_statement(self, items):
        """while_statement: "while" "(" boolean_expression ")" "do" block"""
        condition = self._to_node(items[0])
        body = items[1]
        return WhileNode(condition=condition, body=body)
    
    def repeat_statement(self, items):
        """repeat_statement: "repeat" statement+ "until" "(" boolean_expression ")"  """
        statements = items[:-1]
        condition = self._to_node(items[-1])
        body = BlockNode(statements=list(statements))
        return RepeatNode(body=body, condition=condition)
    
    def if_statement(self, items):
        """if_statement: "if" "(" boolean_expression ")" "then" block ("else" block)?"""
        condition = self._to_node(items[0])
        then_block = items[1]
        else_block = items[2] if len(items) > 2 else None
        return IfNode(condition=condition, then_block=then_block, else_block=else_block)
    
    def assignment(self, items):
        """assignment: lvalue "←" expression"""
        target = items[0]  # Ya es un LValueNode
        value = self._to_node(items[1])
        
        # CRÍTICO: Asegurar que target es un LValueNode
        if isinstance(target, Token):
            target = VariableLValueNode(name=target.value)
        elif isinstance(target, IdentifierNode):
            target = VariableLValueNode(name=target.name)
        
        return AssignmentNode(target=target, value=value)
    
    def call_statement(self, items):
        """call_statement: "call" IDENTIFIER "(" argument_list? ")"  """
        name = self._extract_name(items[0])
        arguments = items[1] if len(items) > 1 and items[1] is not None else []
        
        # Convertir argumentos
        if isinstance(arguments, list):
            arguments = [self._to_node(arg) for arg in arguments]
        
        return CallStatementNode(name=name, arguments=arguments)
    
    def return_statement(self, items):
        """return_statement: "return" expression?"""
        value = self._to_node(items[0]) if items else None
        return ReturnNode(value=value)
    
    def argument_list(self, items):
        """argument_list: expression ("," expression)*"""
        return [self._to_node(item) for item in items]
    
    # ========================================================================
    # LVALUES (CON ALIAS)
    # ========================================================================
    
    def variable_lvalue(self, items):
        """variable_lvalue: IDENTIFIER"""
        name = self._extract_name(items[0])
        return VariableLValueNode(name=name)
    
    def array_object_lvalue(self, items):
        """array_object_lvalue: IDENTIFIER ("[" expression "]")+ ("." IDENTIFIER)*"""
        name = self._extract_name(items[0])
        
        indices = []
        fields = []
        
        for item in items[1:]:
            if isinstance(item, Token) and item.type == 'IDENTIFIER':
                fields.append(item.value)
            elif isinstance(item, str):
                fields.append(item)
            else:
                indices.append(self._to_node(item))
        
        if fields:
            # Tiene campos - retornar array con acceso a objeto (raro)
            return ArrayLValueNode(name=name, indices=indices)
        else:
            return ArrayLValueNode(name=name, indices=indices)
    
    def object_lvalue(self, items):
        """object_lvalue: IDENTIFIER ("." IDENTIFIER)+"""
        name = self._extract_name(items[0])
        fields = [self._extract_name(item) for item in items[1:]]
        return ObjectLValueNode(object_name=name, fields=fields)
    
    # ========================================================================
    # EXPRESIONES: OPERADORES
    # ========================================================================
    
    def add(self, items):
        return BinaryOpNode(op="+", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def sub(self, items):
        return BinaryOpNode(op="-", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def mul(self, items):
        return BinaryOpNode(op="*", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def div(self, items):
        return BinaryOpNode(op="/", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def mod(self, items):
        return BinaryOpNode(op="mod", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def int_div(self, items):
        return BinaryOpNode(op="div", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def pow(self, items):
        return BinaryOpNode(op="^", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def pos(self, items):
        return UnaryOpNode(op="+", operand=self._to_node(items[0]))
    
    def neg(self, items):
        return UnaryOpNode(op="-", operand=self._to_node(items[0]))
    
    def lt(self, items):
        return BinaryOpNode(op="<", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def gt(self, items):
        return BinaryOpNode(op=">", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def le(self, items):
        return BinaryOpNode(op="≤", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def ge(self, items):
        return BinaryOpNode(op="≥", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def eq(self, items):
        return BinaryOpNode(op="=", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def ne(self, items):
        return BinaryOpNode(op="≠", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def and_op(self, items):
        return BinaryOpNode(op="and", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def or_op(self, items):
        return BinaryOpNode(op="or", left=self._to_node(items[0]), right=self._to_node(items[1]))
    
    def not_op(self, items):
        return UnaryOpNode(op="not", operand=self._to_node(items[0]))
    
    # ========================================================================
    # EXPRESIONES: PRIMARY (CORREGIDO)
    # ========================================================================
    
    def primary(self, items):
        """
        primary puede ser:
        - NUMBER, STRING, BOOLEAN, NULL
        - IDENTIFIER
        - IDENTIFIER ("[" array_index "]")+
        - IDENTIFIER ("." IDENTIFIER)+
        - function_call, length, ceiling, floor
        - "(" expression ")"
        """
        if not items:
            return None
        
        # Un solo item - retornar convertido
        if len(items) == 1:
            return self._to_node(items[0])
        
        # Múltiples items - construir nodo compuesto
        first = items[0]
        
        # Si el primer item es un nodo complejo, retornarlo
        if isinstance(first, (FunctionCallNode, LengthNode, CeilingNode, FloorNode)):
            return first
        
        # Extraer nombre base
        name = self._extract_name(first)
        
        # Clasificar items restantes
        indices = []
        fields = []
        
        for item in items[1:]:
            if isinstance(item, Token):
                if item.type == 'IDENTIFIER':
                    fields.append(item.value)
                else:
                    indices.append(self._to_node(item))
            elif isinstance(item, str):
                fields.append(item)
            elif isinstance(item, (RangeNode, NumberNode, IdentifierNode, BinaryOpNode)):
                indices.append(item)
            else:
                converted = self._to_node(item)
                if isinstance(converted, str):
                    fields.append(converted)
                else:
                    indices.append(converted)
        
        # Construir nodo apropiado
        if indices and not fields:
            return ArrayAccessNode(name=name, indices=indices)
        elif fields and not indices:
            return ObjectAccessNode(object_name=name, fields=fields)
        else:
            return IdentifierNode(name=name)
    
    # ========================================================================
    # EXPRESIONES: FUNCIONES Y VALORES
    # ========================================================================
    
    def array_index(self, items):
        """array_index: expression | expression ".." expression"""
        if len(items) == 1:
            return self._to_node(items[0])
        else:
            return RangeNode(start=self._to_node(items[0]), end=self._to_node(items[1]))
    
    def function_call(self, items):
        """function_call: "call" IDENTIFIER "(" argument_list? ")"  """
        name = self._extract_name(items[0])
        arguments = items[1] if len(items) > 1 and items[1] is not None else []
        
        if isinstance(arguments, list):
            arguments = [self._to_node(arg) for arg in arguments]
        
        return FunctionCallNode(name=name, arguments=arguments)
    
    def length_function(self, items):
        """length_function: "length" "(" IDENTIFIER ")"  """
        array_name = self._extract_name(items[0])
        return LengthNode(array_name=array_name)
    
    def ceiling_function(self, items):
        """ceiling_function: "┌" expression "┐" | "ceil" "(" expression ")"  """
        return CeilingNode(expression=self._to_node(items[0]))
    
    def floor_function(self, items):
        """floor_function: "└" expression "┘" | "floor" "(" expression ")"  """
        return FloorNode(expression=self._to_node(items[0]))


# ============================================================================
# FUNCIÓN HELPER
# ============================================================================

def transform_to_ast(lark_tree) -> ProgramNode:
    """
    Transforma un árbol de Lark en nuestro AST personalizado.
    
    Args:
        lark_tree: Árbol generado por el parser de Lark
        
    Returns:
        ProgramNode: Raíz del AST personalizado
    """
    transformer = PseudocodeTransformer(visit_tokens=False)
    return transformer.transform(lark_tree)


# ============================================================================
# TEST RÁPIDO
# ============================================================================

if __name__ == "__main__":
    from lark import Lark
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grammar_path = os.path.join(script_dir, '../parser', 'grammar.lark')
    
    with open(grammar_path, 'r', encoding='utf-8') as f:
        grammar = f.read()
    
    parser = Lark(grammar, parser='earley', start='program')
    
    # Test simple
    test_code = """
Simple()
begin
    x ← 5
    y ← x + 3
end
    """
    
    print("="*70)
    print("TEST DEL TRANSFORMER CORREGIDO")
    print("="*70)
    
    try:
        lark_tree = parser.parse(test_code)
        ast = transform_to_ast(lark_tree)
        
        print("✓ Transformación exitosa")
        print(f"\nProcedimiento: {ast.procedures[0].name}")
        print(f"Statements: {len(ast.procedures[0].body.statements)}")
        
        # Verificar que no hay Tokens
        def check_no_tokens(node, path="root"):
            if isinstance(node, Token):
                print(f"✗ ERROR: Token encontrado en {path}")
                return False
            
            if hasattr(node, '__dict__'):
                for attr, value in node.__dict__.items():
                    if isinstance(value, Token):
                        print(f"✗ ERROR: Token en {path}.{attr}")
                        return False
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if not check_no_tokens(item, f"{path}.{attr}[{i}]"):
                                return False
                    elif hasattr(value, '__dict__'):
                        if not check_no_tokens(value, f"{path}.{attr}"):
                            return False
            return True
        
        if check_no_tokens(ast):
            print("✓ Sin Tokens residuales")
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()