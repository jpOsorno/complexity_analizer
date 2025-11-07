# nodes.py
# Definición del Árbol de Sintaxis Abstracta (AST) tipado
# Basado en el árbol sintáctico producido por el parser

from typing import List, Optional, Any


# ----------------------------
#  Nodo base
# ----------------------------
class Node:
    """Nodo base del AST tipado"""

    def __init__(self, nodetype: str):
        self.type = nodetype

    def to_dict(self) -> dict:
        """Convierte el nodo en un dict para inspección/serialización"""
        result = {"type": self.type}
        for k, v in self.__dict__.items():
            if k != "type" and not k.startswith("_"):
                if isinstance(v, Node):
                    result[k] = v.to_dict()
                elif isinstance(v, list):
                    result[k] = [x.to_dict() if isinstance(x, Node) else x for x in v]
                else:
                    result[k] = v
        return result

    def __repr__(self):
        return f"{self.type}({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items() if k != 'type')})"


# ----------------------------
#  Nodos concretos
# ----------------------------

class ProgramNode(Node):
    def __init__(self, body: List[Node]):
        super().__init__("program")
        self.body = body


class ForNode(Node):
    def __init__(self, var: str, start: Any, end: Any, body: List[Node]):
        super().__init__("for")
        self.var = var
        self.start = start
        self.end = end
        self.body = body


class WhileNode(Node):
    def __init__(self, cond: Any, body: List[Node]):
        super().__init__("while")
        self.cond = cond
        self.body = body


class IfNode(Node):
    def __init__(self, cond: Any, then_body: List[Node], else_body: Optional[List[Node]] = None):
        super().__init__("if")
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body


class AssignNode(Node):
    def __init__(self, target: Any, expr: Any):
        super().__init__("assign")
        self.target = target
        self.expr = expr


class CallNode(Node):
    def __init__(self, name: str, args: List[Any]):
        super().__init__("call")
        self.name = name
        self.args = args


class VariableNode(Node):
    def __init__(self, name: str):
        super().__init__("var")
        self.name = name


class ArrayAccessNode(Node):
    def __init__(self, name: str, index: Any):
        super().__init__("array_access")
        self.name = name
        self.index = index


class BinaryOpNode(Node):
    def __init__(self, op: str, left: Any, right: Any):
        super().__init__("binop")
        self.op = op
        self.left = left
        self.right = right


class ConstantNode(Node):
    def __init__(self, value: Any):
        super().__init__("const")
        self.value = value


class NullNode(Node):
    def __init__(self):
        super().__init__("null")


# ----------------------------
#  Constructor de AST tipado
# ----------------------------
def build_typed_ast(ast_dict: dict) -> Node:
    """Convierte un ASTNode (dict) del parser en nodos tipados"""

    t = ast_dict.get("type")

    # Programa
    if t == "program":
        return ProgramNode([build_typed_ast(n) for n in ast_dict["body"]])

    # For
    if t == "for":
        return ForNode(
            var=ast_dict["var"],
            start=ast_dict["start"],
            end=ast_dict["end"],
            body=[build_typed_ast(n) for n in ast_dict.get("body", [])],
        )

    # Assign
    if t == "assign":
        return AssignNode(
            target=build_typed_ast(ast_dict["target"]),
            expr=build_typed_ast(ast_dict["expr"]),
        )

    # Variable
    if t == "var":
        return VariableNode(ast_dict["name"])

    # Array access
    if t == "array_access":
        return ArrayAccessNode(name=ast_dict["name"], index=ast_dict["index"])

    # Call
    if t == "call":
        return CallNode(name=ast_dict["name"], args=ast_dict.get("args", []))

    # Binary operation
    if t == "binop":
        return BinaryOpNode(
            op=ast_dict["op"],
            left=build_typed_ast(ast_dict["left"]),
            right=build_typed_ast(ast_dict["right"]),
        )

    # Constantes
    if t == "const" or isinstance(ast_dict, (int, float, str)):
        return ConstantNode(ast_dict.get("value", ast_dict))

    # Null
    if t == "null":
        return NullNode()

    # Por defecto: devuelve un nodo genérico
    return ConstantNode(ast_dict)
