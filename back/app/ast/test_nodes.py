from back.app.ast.nodes import build_typed_ast, ProgramNode
from back.app.parser.parser import parse_code, to_dict


sample = """
for i 🡨 1 to n do
    begin
        A[i] 🡨 A[i] + 1
    end
CALL sumar(A)
"""

# Generar AST genérico (dict)
ast_raw = to_dict(parse_code(sample))

# Convertir a AST tipado
ast_typed = build_typed_ast(ast_raw)

print(ast_typed)
print("\nAST tipado en formato dict:\n", ast_typed.to_dict())
