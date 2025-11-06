# parser.py (versión corregida)
from lark import Lark, Transformer, v_args, Tree, Token
import os
import json

GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")

with open(GRAMMAR_PATH, "r", encoding="utf-8") as f:
    GRAMMAR = f.read()

parser = Lark(GRAMMAR, start="start", parser="lalr", propagate_positions=True, maybe_placeholders=False)


class ASTNode(dict):
    def __init__(self, nodetype, **kwargs):
        super().__init__(type=nodetype, **kwargs)

    def __repr__(self):
        # mostrar compactamente
        d = {k: v for k, v in self.items() if k != "type"}
        return f"ASTNode({self['type']}, {d})"


@v_args(inline=True)
class TreeToAST(Transformer):
    # tokens
    def NAME(self, tok):
        return str(tok)

    def INT(self, tok):
        return int(tok)

    # top-level
    def START(self, *items):
        return ASTNode("program", body=list(items))

    def program(self, *items):
        # flatten
        return ASTNode("program", body=list(items))

    # statements
    def assignment(self, lvalue, expr):
        return ASTNode("assign", target=lvalue, expr=expr)

    def lvalue(self, name, *rest):
        node = ASTNode("var", name=name)
        for part in rest:
            # part can be index node or string (field name)
            if isinstance(part, dict) and part.get("type", "").startswith("index"):
                node.setdefault("indexes", []).append(part)
            elif isinstance(part, str):
                node = ASTNode("field", obj=node, field=part)
            else:
                node.setdefault("extra", []).append(part)
        return node

    def indexing(self, *items):
        # items: expr  OR expr, '..', expr
        if len(items) == 1:
            return ASTNode("index", value=items[0])
        elif len(items) == 3:
            return ASTNode("index_range", start=items[0], end=items[2])
        else:
            return ASTNode("index", value=list(items))

    def decl_stmt(self, *parts):
        return ASTNode("decl", parts=list(parts))

    def call_stmt(self, name, args=None):
        return ASTNode("call", name=name, args=args or [])

    def arg_list(self, *args):
        return list(args)

    def if_stmt(self, cond, *rest):
        # rest contains then-block and optional else-block (as lists of statements)
        return ASTNode("if", cond=cond, raw_then_else=list(rest))

    def while_stmt(self, cond, *rest):
        return ASTNode("while", cond=cond, body=list(rest))

    def for_stmt(self, var, start, end, *rest):
        return ASTNode("for", var=var, start=start, end=end, body=list(rest))

    def repeat_stmt(self, *rest):
        return ASTNode("repeat", raw=list(rest))

    # expressions
    def logic_or(self, *parts):
        if len(parts) == 1:
            return parts[0]
        return ASTNode("or", terms=list(parts))

    def logic_and(self, *parts):
        if len(parts) == 1:
            return parts[0]
        return ASTNode("and", terms=list(parts))

    def comparison(self, left, *rest):
        if not rest:
            return left
        op = rest[0]
        right = rest[1]
        return ASTNode("comp", op=str(op), left=left, right=right)

    def arith(self, *parts):
        # parts puede ser [expr], [expr, '+', expr], [expr, '+', expr, '-', expr], etc.
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        node = parts[0]
        # Avanzar de dos en dos (op, valor)
        for i in range(1, len(parts) - 1, 2):
            op = parts[i]
            right = parts[i + 1]
            node = ASTNode("binop", op=str(op), left=node, right=right)
        return node

    def term(self, *parts):
        # misma lógica para *, /, div, mod
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        node = parts[0]
        for i in range(1, len(parts) - 1, 2):
            op = parts[i]
            right = parts[i + 1]
            node = ASTNode("binop", op=str(op), left=node, right=right)
        return node

    def factor(self, val):
        return val

    def atom(self, *args):
        # Manejo flexible de atom según cantidad y tipo de hijos
        if len(args) == 1:
            return args[0]  # puede ser INT o NAME
        # Casos: NAME "[" expr "]"
        if len(args) == 3 and args[1] == "[":
            return ASTNode("array_access", name=args[0], index=args[2])
        # Casos: NAME "[" expr ".." expr "]"
        if len(args) == 5 and args[1] == "[" and args[3] == "..":
            return ASTNode("array_range", name=args[0], start=args[2], end=args[4])
        # Caso: NAME "." NAME
        if len(args) == 3 and args[1] == ".":
            return ASTNode("field", obj=args[0], field=args[2])
        # Caso: length(NAME)
        if len(args) == 3 and args[0] == "length":
            return ASTNode("length", name=args[2])
        # Caso: (expr)
        if len(args) == 3 and args[0] == "(":
            return args[1]
        # Caso NULL literal
        if len(args) == 1 and args[0] == "NULL":
            return ASTNode("null")
        # Cualquier otro caso no contemplado
        return ASTNode("atom", value=args)



def parse_code(text: str):
    tree = parser.parse(text)
    ast = TreeToAST().transform(tree)
    # Si aún queda un Tree con data 'start', extraemos su hijo principal
    if hasattr(ast, "data") and ast.data == "start" and len(ast.children) == 1:
        ast = ast.children[0]
    return ast

def to_dict(obj):
    """Convierte cualquier objeto ASTNode, Tree o Token a estructuras JSON-compatibles"""
    if isinstance(obj, ASTNode):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, Tree):
        # Si por alguna razón aún queda un subárbol de Lark
        return {"tree": obj.data, "children": [to_dict(c) for c in obj.children]}
    elif isinstance(obj, Token):
        return str(obj)
    else:
        return obj

if __name__ == "__main__":
    sample = """
    ► Ejemplo simple
    for i 🡨 1 to n do
        begin
            A[i] 🡨 A[i] + 1
        end

    CALL sumar(A)
    """
    ast = parse_code(sample)
    # imprimimos JSON legible
    print(json.dumps(to_dict(ast), indent=2, ensure_ascii=False))

