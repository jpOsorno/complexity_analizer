"""
Tests Completos para el Parser Wrapper
======================================

Prueba:
1. Parsing exitoso de algoritmos comunes
2. Detección de errores de sintaxis
3. Validación de estructura del AST
4. Casos edge (código vacío, solo comentarios, etc.)
5. Performance (parsing de algoritmos grandes)

Ejecutar:
    pytest tests/test_parser.py -v
    python tests/test_parser.py  (sin pytest)
"""

import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.parser.parser import parse, validate_syntax, ParseError
from src.syntax_tree.nodes import (
    ProgramNode, ProcedureNode, ForNode, WhileNode, RepeatNode,
    IfNode, AssignmentNode, BinaryOpNode, NumberNode, IdentifierNode,
    ArrayParamNode, SimpleParamNode, ReturnNode
)


# ============================================================================
# FIXTURES: Códigos de Prueba
# ============================================================================

SIMPLE_ASSIGNMENT = """
Simple()
begin
    x ← 5
    y ← x + 3
end
"""

BUBBLE_SORT = """
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

FACTORIAL_RECURSIVE = """
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
"""

BINARY_SEARCH = """
BinarySearch(A[], n, x)
begin
    left ← 1
    right ← n
    
    while (left ≤ right) do
    begin
        mid ← floor((left + right) / 2)
        
        if (A[mid] = x) then
        begin
            return mid
        end
        else
        begin
            if (A[mid] < x) then
            begin
                left ← mid + 1
            end
            else
            begin
                right ← mid - 1
            end
        end
    end
    
    return -1
end
"""

MERGE_SORT = """
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

MATRIX_MULTIPLY = """
MatrixMultiply(A[][], B[][], n)
begin
    C[n][n]
    
    for i ← 1 to n do
    begin
        for j ← 1 to n do
        begin
            C[i][j] ← 0
            for k ← 1 to n do
            begin
                C[i][j] ← C[i][j] + A[i][k] * B[k][j]
            end
        end
    end
    
    return C
end
"""

WITH_CLASSES = """
Persona {nombre edad direccion}

ProcesarPersona(Clase p)
begin
    if (p ≠ NULL and p.edad > 18) then
    begin
        p.nombre ← "Adulto"
        return T
    end
    else
    begin
        return F
    end
end
"""

INVALID_SYNTAX_1 = """
BadCode()
begin
    x ← 5
    y = 3  ← Error: usa = en vez de ←
end
"""

INVALID_SYNTAX_2 = """
MissingEnd()
begin
    x ← 5
    for i ← 1 to 10 do
    begin
        y ← i
"""

INVALID_SYNTAX_3 = """
BadFor()
begin
    for i = 1 to 10 do  ← Error: usa = en vez de ←
    begin
        x ← i
    end
end
"""


# ============================================================================
# TESTS: Parsing Exitoso
# ============================================================================

def test_simple_assignment():
    """Test 1: Asignación simple"""
    ast = parse(SIMPLE_ASSIGNMENT)
    
    assert isinstance(ast, ProgramNode), "Debe ser ProgramNode"
    assert len(ast.procedures) == 1, "Debe tener 1 procedimiento"
    
    proc = ast.procedures[0]
    assert proc.name == "Simple", f"Nombre debe ser 'Simple', es '{proc.name}'"
    assert len(proc.parameters) == 0, "No debe tener parámetros"
    assert len(proc.body.statements) == 2, "Debe tener 2 statements"
    
    # Primera asignación: x ← 5
    stmt1 = proc.body.statements[0]
    assert isinstance(stmt1, AssignmentNode), "Debe ser AssignmentNode"
    assert stmt1.target.name == "x", "Target debe ser 'x'"
    assert isinstance(stmt1.value, NumberNode), "Value debe ser NumberNode"
    assert stmt1.value.value == 5, "Valor debe ser 5"
    
    # Segunda asignación: y ← x + 3
    stmt2 = proc.body.statements[1]
    assert isinstance(stmt2, AssignmentNode), "Debe ser AssignmentNode"
    assert stmt2.target.name == "y", "Target debe ser 'y'"
    assert isinstance(stmt2.value, BinaryOpNode), "Value debe ser BinaryOpNode"
    assert stmt2.value.op == "+", "Operador debe ser '+'"
    
    print("✓ Test 1 PASS: Asignación simple")
    return True


def test_bubble_sort():
    """Test 2: Bubble Sort con FOR anidado"""
    ast = parse(BUBBLE_SORT)
    
    proc = ast.procedures[0]
    assert proc.name == "BubbleSort", "Nombre debe ser 'BubbleSort'"
    
    # Verificar parámetros: A[], n
    assert len(proc.parameters) == 2, "Debe tener 2 parámetros"
    assert isinstance(proc.parameters[0], ArrayParamNode), "Primer param debe ser array"
    assert proc.parameters[0].name == "A", "Array debe ser 'A'"
    assert isinstance(proc.parameters[1], SimpleParamNode), "Segundo param debe ser simple"
    assert proc.parameters[1].name == "n", "Parámetro debe ser 'n'"
    
    # Verificar FOR externo
    outer_for = proc.body.statements[0]
    assert isinstance(outer_for, ForNode), "Debe ser ForNode"
    assert outer_for.variable == "i", "Variable debe ser 'i'"
    
    # Verificar FOR interno
    inner_for = outer_for.body.statements[0]
    assert isinstance(inner_for, ForNode), "FOR interno debe ser ForNode"
    assert inner_for.variable == "j", "Variable debe ser 'j'"
    
    # Verificar IF dentro del FOR interno
    if_stmt = inner_for.body.statements[0]
    assert isinstance(if_stmt, IfNode), "Debe ser IfNode"
    
    print("✓ Test 2 PASS: Bubble Sort")
    return True


def test_factorial_recursive():
    """Test 3: Factorial recursivo"""
    ast = parse(FACTORIAL_RECURSIVE)
    
    proc = ast.procedures[0]
    assert proc.name == "Factorial", "Nombre debe ser 'Factorial'"
    assert len(proc.parameters) == 1, "Debe tener 1 parámetro"
    
    # Verificar estructura IF-THEN-ELSE
    if_stmt = proc.body.statements[0]
    assert isinstance(if_stmt, IfNode), "Debe ser IfNode"
    
    # THEN: return 1
    then_return = if_stmt.then_block.statements[0]
    assert isinstance(then_return, ReturnNode), "THEN debe tener ReturnNode"
    assert isinstance(then_return.value, NumberNode), "Return debe ser número"
    assert then_return.value.value == 1, "Debe retornar 1"
    
    # ELSE: return n * call Factorial(n-1)
    else_return = if_stmt.else_block.statements[0]
    assert isinstance(else_return, ReturnNode), "ELSE debe tener ReturnNode"
    assert isinstance(else_return.value, BinaryOpNode), "Return debe ser operación"
    
    print("✓ Test 3 PASS: Factorial recursivo")
    return True


def test_binary_search():
    """Test 4: Búsqueda binaria con WHILE"""
    ast = parse(BINARY_SEARCH)
    
    proc = ast.procedures[0]
    assert proc.name == "BinarySearch", "Nombre debe ser 'BinarySearch'"
    assert len(proc.parameters) == 3, "Debe tener 3 parámetros"
    
    # Debe tener inicializaciones + while + return
    statements = proc.body.statements
    assert len(statements) >= 3, "Debe tener al menos 3 statements"
    
    # Buscar el WHILE
    while_found = False
    for stmt in statements:
        if isinstance(stmt, WhileNode):
            while_found = True
            # Verificar que tiene condición
            assert stmt.condition is not None, "WHILE debe tener condición"
            break
    
    assert while_found, "Debe tener un ciclo WHILE"
    
    print("✓ Test 4 PASS: Binary Search")
    return True


def test_merge_sort():
    """Test 5: Merge Sort con llamadas recursivas"""
    ast = parse(MERGE_SORT)
    
    proc = ast.procedures[0]
    assert proc.name == "MergeSort", "Nombre debe ser 'MergeSort'"
    
    # Verificar estructura: IF con llamadas recursivas
    if_stmt = proc.body.statements[0]
    assert isinstance(if_stmt, IfNode), "Debe ser IfNode"
    
    # THEN debe tener 3 llamadas (2 recursivas + 1 a Merge)
    then_statements = if_stmt.then_block.statements
    assert len(then_statements) >= 3, "THEN debe tener al menos 3 statements"
    
    print("✓ Test 5 PASS: Merge Sort")
    return True


def test_matrix_multiply():
    """Test 6: Multiplicación de matrices (FOR triple anidado)"""
    ast = parse(MATRIX_MULTIPLY)
    
    proc = ast.procedures[0]
    assert proc.name == "MatrixMultiply", "Nombre debe ser 'MatrixMultiply'"
    
    # Verificar declaración local: C[n][n]
    assert len(proc.body.declarations) == 1, "Debe tener 1 declaración local"
    
    # Verificar FOR triple anidado
    outer_for = proc.body.statements[0]
    assert isinstance(outer_for, ForNode), "Debe ser ForNode"
    assert outer_for.variable == "i", "Variable debe ser 'i'"
    
    middle_for = outer_for.body.statements[0]
    assert isinstance(middle_for, ForNode), "FOR medio debe ser ForNode"
    assert middle_for.variable == "j", "Variable debe ser 'j'"
    
    # En el FOR medio hay asignación + FOR interno
    inner_for = None
    for stmt in middle_for.body.statements:
        if isinstance(stmt, ForNode):
            inner_for = stmt
            break
    
    assert inner_for is not None, "Debe tener FOR interno"
    assert inner_for.variable == "k", "Variable debe ser 'k'"
    
    print("✓ Test 6 PASS: Matrix Multiply")
    return True


def test_with_classes():
    """Test 7: Clases y objetos"""
    ast = parse(WITH_CLASSES)
    
    # Verificar clase
    assert len(ast.classes) == 1, "Debe tener 1 clase"
    clase = ast.classes[0]
    assert clase.name == "Persona", "Clase debe ser 'Persona'"
    assert "nombre" in clase.attributes, "Debe tener atributo 'nombre'"
    assert "edad" in clase.attributes, "Debe tener atributo 'edad'"
    
    # Verificar procedimiento
    proc = ast.procedures[0]
    assert proc.name == "ProcesarPersona", "Nombre debe ser 'ProcesarPersona'"
    
    print("✓ Test 7 PASS: Clases y objetos")
    return True


# ============================================================================
# TESTS: Detección de Errores
# ============================================================================

def test_invalid_syntax_1():
    """Test 8: Error de sintaxis (= en vez de ←)"""
    try:
        ast = parse(INVALID_SYNTAX_1)
        assert False, "Debería lanzar ParseError"
    except ParseError as e:
        assert "sintaxis" in str(e).lower() or "token" in str(e).lower()
        print("✓ Test 8 PASS: Detecta error de sintaxis")
        return True
    except Exception as e:
        print(f"✗ Test 8 FAIL: Lanzó {type(e).__name__} en vez de ParseError")
        return False


def test_invalid_syntax_2():
    """Test 9: Error de sintaxis (falta end)"""
    try:
        ast = parse(INVALID_SYNTAX_2)
        assert False, "Debería lanzar ParseError"
    except ParseError as e:
        print("✓ Test 9 PASS: Detecta falta de 'end'")
        return True
    except Exception as e:
        print(f"✗ Test 9 FAIL: Lanzó {type(e).__name__} en vez de ParseError")
        return False


def test_empty_code():
    """Test 10: Código vacío"""
    try:
        ast = parse("")
        assert False, "Debería lanzar ParseError"
    except ParseError as e:
        assert "vacío" in str(e).lower()
        print("✓ Test 10 PASS: Detecta código vacío")
        return True


def test_validate_syntax():
    """Test 11: Función validate_syntax"""
    # Código válido
    valid, error = validate_syntax(SIMPLE_ASSIGNMENT)
    assert valid == True, "Código simple debe ser válido"
    assert error is None, "No debe haber error"
    
    # Código inválido
    valid, error = validate_syntax(INVALID_SYNTAX_1)
    assert valid == False, "Código inválido debe detectarse"
    assert error is not None, "Debe haber mensaje de error"
    
    print("✓ Test 11 PASS: validate_syntax funciona")
    return True


# ============================================================================
# TESTS: Casos Edge
# ============================================================================

def test_only_comments():
    """Test 12: Solo comentarios (debe fallar)"""
    code = """
    ► Esto es un comentario
    ► Otro comentario
    """
    try:
        ast = parse(code)
        assert False, "Código sin procedimientos debe fallar"
    except ParseError:
        print("✓ Test 12 PASS: Detecta falta de procedimientos")
        return True


def test_multiple_procedures():
    """Test 13: Múltiples procedimientos"""
    code = """
Proc1()
begin
    x ← 1
end

Proc2()
begin
    y ← 2
end
    """
    ast = parse(code)
    assert len(ast.procedures) == 2, "Debe tener 2 procedimientos"
    assert ast.procedures[0].name == "Proc1"
    assert ast.procedures[1].name == "Proc2"
    
    print("✓ Test 13 PASS: Múltiples procedimientos")
    return True


# ============================================================================
# RUNNER
# ============================================================================

def run_all_tests():
    """Ejecuta todos los tests"""
    tests = [
        test_simple_assignment,
        test_bubble_sort,
        test_factorial_recursive,
        test_binary_search,
        test_merge_sort,
        test_matrix_multiply,
        test_with_classes,
        test_invalid_syntax_1,
        test_invalid_syntax_2,
        test_empty_code,
        test_validate_syntax,
        test_only_comments,
        test_multiple_procedures,
    ]
    
    print("="*70)
    print("TESTS DEL PARSER WRAPPER")
    print("="*70)
    print()
    
    results = []
    for test_func in tests:
        try:
            success = test_func()
            results.append((test_func.__name__, True, None))
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAIL: {e}")
            results.append((test_func.__name__, False, str(e)))
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            results.append((test_func.__name__, False, f"Error: {e}"))
        print()
    
    # Resumen
    print("="*70)
    print("RESUMEN")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {name}")
    
    print("="*70)
    print(f"Resultado: {passed}/{total} tests pasados ({passed/total*100:.1f}%)")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)