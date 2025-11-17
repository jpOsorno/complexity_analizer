"""
Transformer: Convierte árbol de Lark a AST personalizado (VERSIÓN FINAL CORREGIDA)
===================================================================================

Fix crítico: Asegurar que NO queden objetos Tree o Token sin transformar
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
    
    def __init__(self, *args, **kwargs):
        # CRÍTICO: visit_tokens=True para transformar tokens automáticamente
        super().__init__(visit_tokens=True, *args, **kwargs)
    
    # ========================================================================
    # TRANSFORMACIÓN DE TOKENS (llamados automáticamente)
    # ========================================================================
    
    def NUMBER(self, token):
        """Transforma token NUMBER a NumberNode"""
        value = token.value
        return NumberNode(value=float(value) if '.' in value else int(value))
    
    def IDENTIFIER(self, token):
        """Transforma token IDENTIFIER a IdentifierNode"""
        return IdentifierNode(name=token.value)
    
    def STRING(self, token):
        """Transforma token STRING a StringNode"""
        content = token.value[1:-1] if len(token.value) >= 2 else token.value
        return StringNode(value=content)
    
    def BOOLEAN(self, token):
        """Transforma token BOOLEAN a BooleanNode"""
        return BooleanNode(value=token.value in ('T', 'true'))
    
    # ========================================================================
    # HELPERS MEJORADOS
    # ========================================================================
    
    @staticmethod
    def _ensure_node(item):
        """
        Garantiza que item sea un nodo del AST (no Tree ni Token).
        
        CRÍTICO: Esta función previene errores de Tree/Token sin transformar.
        """
        # Ya es un nodo del AST - OK
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
        
        # ERROR: Tree no transformado
        if isinstance(item, Tree):
            raise ValueError(
                f"Tree sin transformar encontrado: {item.data}. "
                f"Falta método transformer para esta regla."
            )
        
        # ERROR: Token no transformado (no debería pasar con visit_tokens=True)
        if isinstance(item, Token):
            # Último recurso: convertir a nodo apropiado
            if item.type == 'NUMBER':
                value = item.value
                return NumberNode(value=float(value) if '.' in value else int(value))
            elif item.type == 'IDENTIFIER':
                return IdentifierNode(name=item.value)
            elif item.type == 'STRING':
                content = item.value[1:-1] if len(item.value) >= 2 else item.value
                return StringNode(value=content)
            elif item.type == 'BOOLEAN':
                return BooleanNode(value=item.value in ('T', 'true'))
            else:
                raise ValueError(f"Token inesperado: {item.type} = {item.value}")
        
        # Valores primitivos
        if isinstance(item, (int, float)):
            return NumberNode(value=item)
        if isinstance(item, str):
            return IdentifierNode(name=item)
        if isinstance(item, bool):
            return BooleanNode(value=item)
        
        # None
        if item is None:
            return None
        
        # Lista - transformar cada elemento
        if isinstance(item, list):
            return [PseudocodeTransformer._ensure_node(x) for x in item]
        
        # Tipo desconocido
        raise ValueError(f"Tipo no soportado en _ensure_node: {type(item).__name__}")
    
    @staticmethod
    def _extract_name(item) -> str:
        """Extrae nombre de string de cualquier item"""
        if isinstance(item, IdentifierNode):
            return item.name
        elif isinstance(item, str):
            return item
        elif isinstance(item, Token):
            return item.value
        else:
            raise ValueError(f"No se puede extraer nombre de {type(item).__name__}")
    
    # ========================================================================
    # PROGRAMA Y DEFINICIONES
    # ========================================================================
    
    def program(self, items):
        """program: (class_definition)* procedure_definition+"""
        classes = []
        procedures = []
        
        for item in items:
            item = self._ensure_node(item)
            if isinstance(item, ClassDefNode):
                classes.append(item)
            elif isinstance(item, ProcedureNode):
                procedures.append(item)
        
        return ProgramNode(classes=classes, procedures=procedures)
    
    def class_definition(self, items):
        """class_definition: IDENTIFIER "{" attribute_list "}"  """
        name = self._extract_name(self._ensure_node(items[0]))
        attributes = items[1] if len(items) > 1 else []
        return ClassDefNode(name=name, attributes=attributes)
    
    def attribute_list(self, items):
        """attribute_list: IDENTIFIER+"""
        return [self._extract_name(self._ensure_node(item)) for item in items]
    
    def procedure_definition(self, items):
        """procedure_definition: IDENTIFIER "(" parameter_list? ")" block"""
        name = self._extract_name(self._ensure_node(items[0]))
        
        if len(items) == 2:
            parameters = []
            body = self._ensure_node(items[1])
        else:
            parameters = items[1] if isinstance(items[1], list) else []
            body = self._ensure_node(items[2])
        
        return ProcedureNode(name=name, parameters=parameters, body=body)
    
    def parameter_list(self, items):
        """parameter_list: parameter ("," parameter)*"""
        return [self._ensure_node(item) for item in items]
    
    def parameter(self, items):
        """parameter: class_param | array_param | simple_param"""
        return self._ensure_node(items[0])
    
    def simple_param(self, items):
        """simple_param: IDENTIFIER"""
        name = self._extract_name(self._ensure_node(items[0]))
        return SimpleParamNode(name=name)
    
    def array_param(self, items):
        """array_param: IDENTIFIER ("[" NUMBER? "]")+"""
        name = self._extract_name(self._ensure_node(items[0]))
        dimensions = []
        
        for item in items[1:]:
            if item is None:
                dimensions.append(None)
            else:
                node = self._ensure_node(item)
                if isinstance(node, NumberNode):
                    dimensions.append(node.value)
                else:
                    dimensions.append(item)
        
        return ArrayParamNode(name=name, dimensions=dimensions)
    
    def class_param(self, items):
        """class_param: "Clase" IDENTIFIER"""
        name = self._extract_name(self._ensure_node(items[-1]))
        return ClassParamNode(name=name, class_name="Clase")
    
    # ========================================================================
    # BLOQUES Y DECLARACIONES
    # ========================================================================
    
    def block(self, items):
        """block: "begin" declaration* statement* "end" """
        declarations = []
        statements = []
        
        for item in items:
            if item is None:
                continue
            
            node = self._ensure_node(item)
            
            if isinstance(node, (ArrayDeclNode, ObjectDeclNode)):
                declarations.append(node)
            else:
                statements.append(node)
        
        return BlockNode(declarations=declarations, statements=statements)
    
    def local_array_decl(self, items):
        """local_array_decl: IDENTIFIER ("[" expression? "]")+"""
        name = self._extract_name(self._ensure_node(items[0]))
        dimensions = [self._ensure_node(item) for item in items[1:]]
        return ArrayDeclNode(name=name, dimensions=dimensions)
    
    def local_object_decl(self, items):
        """local_object_decl: "Clase" IDENTIFIER"""
        name = self._extract_name(self._ensure_node(items[-1]))
        return ObjectDeclNode(class_name="Clase", object_name=name)
    
    # ========================================================================
    # SENTENCIAS
    # ========================================================================
    
    def for_statement(self, items):
        """for_statement: "for" IDENTIFIER "←" expression "to" expression "do" block"""
        variable = self._extract_name(self._ensure_node(items[0]))
        start = self._ensure_node(items[1])
        end = self._ensure_node(items[2])
        body = self._ensure_node(items[3])
        
        return ForNode(variable=variable, start=start, end=end, body=body)
    
    def while_statement(self, items):
        """while_statement: "while" "(" boolean_expression ")" "do" block"""
        condition = self._ensure_node(items[0])
        body = self._ensure_node(items[1])
        return WhileNode(condition=condition, body=body)
    
    def repeat_statement(self, items):
        """repeat_statement: "repeat" statement+ "until" "(" boolean_expression ")"  """
        statements = [self._ensure_node(item) for item in items[:-1]]
        condition = self._ensure_node(items[-1])
        body = BlockNode(statements=statements)
        return RepeatNode(body=body, condition=condition)
    
    def if_statement(self, items):
        """if_statement: "if" "(" boolean_expression ")" "then" block ("else" block)?"""
        condition = self._ensure_node(items[0])
        then_block = self._ensure_node(items[1])
        else_block = self._ensure_node(items[2]) if len(items) > 2 else None
        return IfNode(condition=condition, then_block=then_block, else_block=else_block)
    
    def assignment(self, items):
        """assignment: lvalue "←" expression"""
        target = self._ensure_node(items[0])
        value = self._ensure_node(items[1])
        
        # Asegurar que target es un LValueNode
        if isinstance(target, IdentifierNode):
            target = VariableLValueNode(name=target.name)
        
        return AssignmentNode(target=target, value=value)
    
    def call_statement(self, items):
        """call_statement: "call" IDENTIFIER "(" argument_list? ")"  """
        name = self._extract_name(self._ensure_node(items[0]))
        arguments = []
        
        if len(items) > 1 and items[1] is not None:
            args = items[1]
            if isinstance(args, list):
                arguments = [self._ensure_node(arg) for arg in args]
            else:
                arguments = [self._ensure_node(args)]
        
        return CallStatementNode(name=name, arguments=arguments)
    
    def return_statement(self, items):
        """return_statement: "return" expression?"""
        value = self._ensure_node(items[0]) if items else None
        return ReturnNode(value=value)
    
    def argument_list(self, items):
        """argument_list: expression ("," expression)*"""
        return [self._ensure_node(item) for item in items]
    
    # ========================================================================
    # LVALUES
    # ========================================================================
    
    def variable_lvalue(self, items):
        """variable_lvalue: IDENTIFIER"""
        name = self._extract_name(self._ensure_node(items[0]))
        return VariableLValueNode(name=name)
    
    def array_object_lvalue(self, items):
        """array_object_lvalue: IDENTIFIER ("[" expression "]")+ ("." IDENTIFIER)*"""
        name = self._extract_name(self._ensure_node(items[0]))
        
        indices = []
        fields = []
        
        for item in items[1:]:
            node = self._ensure_node(item)
            if isinstance(node, IdentifierNode):
                fields.append(node.name)
            elif isinstance(node, str):
                fields.append(node)
            else:
                indices.append(node)
        
        return ArrayLValueNode(name=name, indices=indices)
    
    def object_lvalue(self, items):
        """object_lvalue: IDENTIFIER ("." IDENTIFIER)+"""
        name = self._extract_name(self._ensure_node(items[0]))
        fields = [self._extract_name(self._ensure_node(item)) for item in items[1:]]
        return ObjectLValueNode(object_name=name, fields=fields)
    
    # ========================================================================
    # EXPRESIONES: OPERADORES
    # ========================================================================
    
    def add(self, items):
        return BinaryOpNode(op="+", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def sub(self, items):
        return BinaryOpNode(op="-", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def mul(self, items):
        return BinaryOpNode(op="*", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def div(self, items):
        return BinaryOpNode(op="/", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def mod(self, items):
        return BinaryOpNode(op="mod", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def int_div(self, items):
        return BinaryOpNode(op="div", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def pow(self, items):
        return BinaryOpNode(op="^", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def pos(self, items):
        return UnaryOpNode(op="+", operand=self._ensure_node(items[0]))
    
    def neg(self, items):
        return UnaryOpNode(op="-", operand=self._ensure_node(items[0]))
    
    def lt(self, items):
        return BinaryOpNode(op="<", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def gt(self, items):
        return BinaryOpNode(op=">", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def le(self, items):
        return BinaryOpNode(op="≤", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def ge(self, items):
        return BinaryOpNode(op="≥", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def eq(self, items):
        return BinaryOpNode(op="=", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def ne(self, items):
        return BinaryOpNode(op="≠", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def and_op(self, items):
        return BinaryOpNode(op="and", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def or_op(self, items):
        return BinaryOpNode(op="or", 
                          left=self._ensure_node(items[0]), 
                          right=self._ensure_node(items[1]))
    
    def not_op(self, items):
        return UnaryOpNode(op="not", operand=self._ensure_node(items[0]))
    
    # ========================================================================
    # EXPRESIONES: PRIMARY
    # ========================================================================
    
    def identifier(self, items):
        """identifier: IDENTIFIER"""
        return self._ensure_node(items[0])
    
    def null_literal(self, items):
        """null_literal: "NULL" """
        return NullNode()
    
    def array_object_access(self, items):
        """array_object_access: IDENTIFIER ("[" array_index "]")+ ("." IDENTIFIER)*"""
        name = self._extract_name(self._ensure_node(items[0]))
        
        indices = []
        fields = []
        
        for item in items[1:]:
            node = self._ensure_node(item)
            if isinstance(node, IdentifierNode):
                fields.append(node.name)
            elif isinstance(node, str):
                fields.append(node)
            else:
                indices.append(node)
        
        return ArrayAccessNode(name=name, indices=indices)
    
    def object_access(self, items):
        """object_access: IDENTIFIER ("." IDENTIFIER)+"""
        name = self._extract_name(self._ensure_node(items[0]))
        fields = [self._extract_name(self._ensure_node(item)) for item in items[1:]]
        return ObjectAccessNode(object_name=name, fields=fields)
    
    # ========================================================================
    # EXPRESIONES: FUNCIONES
    # ========================================================================
    
    def array_index(self, items):
        """array_index: expression | expression ".." expression"""
        if len(items) == 1:
            return self._ensure_node(items[0])
        else:
            return RangeNode(
                start=self._ensure_node(items[0]), 
                end=self._ensure_node(items[1])
            )
    
    def function_call(self, items):
        """function_call: "call" IDENTIFIER "(" argument_list? ")"  """
        name = self._extract_name(self._ensure_node(items[0]))
        arguments = []
        
        if len(items) > 1 and items[1] is not None:
            args = items[1]
            if isinstance(args, list):
                arguments = [self._ensure_node(arg) for arg in args]
            else:
                arguments = [self._ensure_node(args)]
        
        return FunctionCallNode(name=name, arguments=arguments)
    
    def length_function(self, items):
        """length_function: "length" "(" IDENTIFIER ")"  """
        name = self._extract_name(self._ensure_node(items[0]))
        return LengthNode(array_name=name)
    
    def ceiling_function(self, items):
        """ceiling_function: "┌" expression "┐" | "ceil" "(" expression ")"  """
        return CeilingNode(expression=self._ensure_node(items[0]))
    
    def floor_function(self, items):
        """floor_function: "└" expression "┘" | "floor" "(" expression ")"  """
        return FloorNode(expression=self._ensure_node(items[0]))


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
    transformer = PseudocodeTransformer()
    ast = transformer.transform(lark_tree)
    
    # Validación final: asegurar que no hay Tree ni Token
    def validate_ast(node, path="root"):
        if isinstance(node, Tree):
            raise ValueError(f"Tree sin transformar en {path}: {node.data}")
        if isinstance(node, Token):
            raise ValueError(f"Token sin transformar en {path}: {node.type}={node.value}")
        
        if hasattr(node, '__dict__'):
            for attr, value in node.__dict__.items():
                if isinstance(value, (Tree, Token)):
                    raise ValueError(f"Tree/Token en {path}.{attr}")
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        validate_ast(item, f"{path}.{attr}[{i}]")
                elif hasattr(value, '__dict__'):
                    validate_ast(value, f"{path}.{attr}")
    
    validate_ast(ast)
    return ast


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    from lark import Lark
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grammar_path = os.path.join(script_dir, '../parser', 'grammar.lark')
    
    with open(grammar_path, 'r', encoding='utf-8') as f:
        grammar = f.read()
    
    parser = Lark(grammar, parser='earley', start='program')
    
    test_code = """
Simplified(A[], n)
begin
    for i ← 1 to n-1 do
    begin
        for j ← i to n-i do
        begin
            x ← A[i] + A[j]
        end
    end
end
    """
    
    print("="*70)
    print("TEST DEL TRANSFORMER")
    print("="*70)
    
    try:
        lark_tree = parser.parse(test_code)
        print("\n1. Árbol de Lark:")
        print(lark_tree.pretty())
        ast = transform_to_ast(lark_tree)

        print("\n2. AST Personalizado:")
        print(ast)
        print("✓ Transformación exitosa")
        print(f"✓ Procedimiento: {ast.procedures[0].name}")
        print(f"Parámetros: {[p.name for p in ast.procedures[0].parameters]}")
        print(f"Tipo de body: {type(ast.procedures[0].body).__name__}")
        print("✓ Sin Tree ni Token residuales")
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()