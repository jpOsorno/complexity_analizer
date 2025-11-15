"""
Test Completo del Sistema de Parsing y AST
==========================================

Este script prueba:
1. Gramática Lark (parsing)
2. Transformación a AST personalizado
3. Estructura correcta de nodos
4. Integridad de datos

Ejecutar: python test_ast_complete.py
"""

from lark import Lark
import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer import transform_to_ast
from nodes import (
    ProgramNode, ProcedureNode, ForNode, WhileNode, IfNode,
    AssignmentNode, BinaryOpNode, NumberNode, IdentifierNode,
    SimpleParamNode, ArrayParamNode, ArrayAccessNode,
    CallStatementNode, ReturnNode, FunctionCallNode, ArrayLValueNode
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

def load_parser():
    """Carga el parser de Lark"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grammar_path = os.path.join(script_dir, '../parser', 'grammar.lark')
    
    with open(grammar_path, 'r', encoding='utf-8') as f:
        grammar = f.read()
    
    return Lark(grammar, parser='earley', start='program')


# ============================================================================
# TESTS
# ============================================================================

def test_simple_assignment():
    """Test 1: Asignación simple"""
    code = """
Simple()
begin
    x ← 5
end
    """
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    # Verificaciones
    assert isinstance(ast, ProgramNode), "Debe ser ProgramNode"
    assert len(ast.procedures) == 1, "Debe tener 1 procedimiento"
    
    proc = ast.procedures[0]
    assert proc.name == "Simple", f"Nombre debe ser 'Simple', es '{proc.name}'"
    assert len(proc.parameters) == 0, "No debe tener parámetros"
    assert len(proc.body.statements) == 1, "Debe tener 1 statement"
    
    stmt = proc.body.statements[0]
    assert isinstance(stmt, AssignmentNode), "Debe ser AssignmentNode"
    assert stmt.target.name == "x", "Target debe ser 'x'"
    assert isinstance(stmt.value, NumberNode), "Value debe ser NumberNode"
    assert stmt.value.value == 5, "Valor debe ser 5"
    
    return True


def test_for_loop():
    """Test 2: Ciclo FOR"""
    code = """
ForTest(n)
begin
    for i ← 1 to n do
    begin
        x ← i * 2
    end
end
    """
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    proc = ast.procedures[0]
    assert proc.name == "ForTest", f"Nombre debe ser 'ForTest'"
    
    # Verificar parámetro
    assert len(proc.parameters) == 1, "Debe tener 1 parámetro"
    assert isinstance(proc.parameters[0], SimpleParamNode), "Debe ser SimpleParamNode"
    assert proc.parameters[0].name == "n", "Parámetro debe ser 'n'"
    
    # Verificar FOR
    for_stmt = proc.body.statements[0]
    assert isinstance(for_stmt, ForNode), "Debe ser ForNode"
    assert for_stmt.variable == "i", "Variable debe ser 'i'"
    assert isinstance(for_stmt.start, NumberNode), "Start debe ser NumberNode"
    assert for_stmt.start.value == 1, "Start debe ser 1"
    assert isinstance(for_stmt.end, IdentifierNode), "End debe ser IdentifierNode"
    assert for_stmt.end.name == "n", "End debe ser 'n'"
    
    # Verificar cuerpo del FOR
    assignment = for_stmt.body.statements[0]
    assert isinstance(assignment, AssignmentNode), "Debe ser AssignmentNode"
    assert isinstance(assignment.value, BinaryOpNode), "Debe ser BinaryOpNode"
    assert assignment.value.op == "*", f"Operador debe ser '*', es '{assignment.value.op}'"
    
    return True


def test_binary_operations():
    """Test 3: Operaciones binarias"""
    code = """
Operations()
begin
    a ← 5 + 3
    b ← 10 - 2
    c ← 4 * 7
    d ← 20 / 5
    e ← 2 ^ 3
end
    """
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    statements = ast.procedures[0].body.statements
    
    # Verificar cada operación
    ops = ['+', '-', '*', '/', '^']
    for i, expected_op in enumerate(ops):
        stmt = statements[i]
        assert isinstance(stmt, AssignmentNode), f"Statement {i} debe ser AssignmentNode"
        assert isinstance(stmt.value, BinaryOpNode), f"Value {i} debe ser BinaryOpNode"
        assert stmt.value.op == expected_op, f"Operador debe ser '{expected_op}', es '{stmt.value.op}'"
    
    return True


def test_nested_for():
    """Test 4: FOR anidado (Bubble Sort simplificado)"""
    code = """
BubbleSort(A[], n)
begin
    for i ← 1 to n-1 do
    begin
        for j ← 1 to n-i do
        begin
            x ← A[j]
        end
    end
end
    """
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    proc = ast.procedures[0]
    
    # Verificar parámetros
    assert len(proc.parameters) == 2, "Debe tener 2 parámetros"
    assert isinstance(proc.parameters[0], ArrayParamNode), "Primer param debe ser ArrayParamNode"
    assert proc.parameters[0].name == "A", "Array debe ser 'A'"
    assert isinstance(proc.parameters[1], SimpleParamNode), "Segundo param debe ser SimpleParamNode"
    assert proc.parameters[1].name == "n", "Parámetro debe ser 'n'"
    
    # Verificar FOR externo
    outer_for = proc.body.statements[0]
    assert isinstance(outer_for, ForNode), "Debe ser ForNode"
    assert outer_for.variable == "i", "Variable debe ser 'i'"
    
    # Verificar FOR interno
    inner_for = outer_for.body.statements[0]
    assert isinstance(inner_for, ForNode), "FOR interno debe ser ForNode"
    assert inner_for.variable == "j", "Variable debe ser 'j'"
    
    # Verificar acceso a array
    assignment = inner_for.body.statements[0]
    assert isinstance(assignment, AssignmentNode), "Debe ser AssignmentNode"
    assert isinstance(assignment.value, ArrayAccessNode), "Debe ser ArrayAccessNode"
    assert assignment.value.name == "A", "Array debe ser 'A'"
    
    return True


def test_while_loop():
    """Test 5: Ciclo WHILE"""
    code = """
WhileTest(n)
begin
    i ← 1
    while (i < n) do
    begin
        i ← i + 1
    end
end
    """
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    statements = ast.procedures[0].body.statements
    
    # Primera statement: asignación
    assert isinstance(statements[0], AssignmentNode), "Primera debe ser AssignmentNode"
    
    # Segunda statement: while
    while_stmt = statements[1]
    assert isinstance(while_stmt, WhileNode), "Debe ser WhileNode"
    assert isinstance(while_stmt.condition, BinaryOpNode), "Condición debe ser BinaryOpNode"
    assert while_stmt.condition.op == "<", f"Operador debe ser '<'"
    
    return True


def test_if_statement():
    """Test 6: Condicional IF-THEN-ELSE"""
    code = """
IfTest(x)
begin
    if (x > 0) then
    begin
        y ← 1
    end
    else
    begin
        y ← -1
    end
end
    """
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    if_stmt = ast.procedures[0].body.statements[0]
    assert isinstance(if_stmt, IfNode), "Debe ser IfNode"
    assert isinstance(if_stmt.condition, BinaryOpNode), "Condición debe ser BinaryOpNode"
    assert if_stmt.condition.op == ">", "Operador debe ser '>'"
    
    # Verificar bloque THEN
    assert len(if_stmt.then_block.statements) == 1, "THEN debe tener 1 statement"
    assert isinstance(if_stmt.then_block.statements[0], AssignmentNode), "THEN debe ser AssignmentNode"
    
    # Verificar bloque ELSE
    assert if_stmt.else_block is not None, "Debe tener ELSE"
    assert len(if_stmt.else_block.statements) == 1, "ELSE debe tener 1 statement"
    
    return True


def test_recursion():
    """Test 7: Recursión (Factorial)"""
    code = """
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
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    proc = ast.procedures[0]
    assert proc.name == "Factorial", "Nombre debe ser 'Factorial'"
    
    if_stmt = proc.body.statements[0]
    
    # Verificar THEN (return 1)
    return_stmt = if_stmt.then_block.statements[0]
    assert isinstance(return_stmt, ReturnNode), "Debe ser ReturnNode"
    assert isinstance(return_stmt.value, NumberNode), "Return debe ser NumberNode"
    
    # Verificar ELSE (return n * call Factorial(n-1))
    return_stmt2 = if_stmt.else_block.statements[0]
    assert isinstance(return_stmt2, ReturnNode), "Debe ser ReturnNode"
    assert isinstance(return_stmt2.value, BinaryOpNode), "Return debe ser BinaryOpNode"
    
    # Verificar llamada recursiva
    mult = return_stmt2.value
    assert isinstance(mult.right, FunctionCallNode), "Debe haber FunctionCallNode"
    assert mult.right.name == "Factorial", "Debe llamar a Factorial"
    
    return True


def test_array_access():
    """Test 8: Acceso a arrays"""
    code = """
ArrayTest(A[], n)
begin
    x ← A[1]
    y ← A[n-1]
    A[5] ← 10
end
    """
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    statements = ast.procedures[0].body.statements
    
    # Primera asignación: x ← A[1]
    stmt1 = statements[0]
    assert isinstance(stmt1.value, ArrayAccessNode), "Debe ser ArrayAccessNode"
    assert stmt1.value.name == "A", "Array debe ser 'A'"
    assert isinstance(stmt1.value.indices[0], NumberNode), "Índice debe ser NumberNode"
    
    # Segunda asignación: y ← A[n-1]
    stmt2 = statements[1]
    assert isinstance(stmt2.value, ArrayAccessNode), "Debe ser ArrayAccessNode"
    assert isinstance(stmt2.value.indices[0], BinaryOpNode), "Índice debe ser BinaryOpNode"
    
    # Tercera asignación: A[5] ← 10
    stmt3 = statements[2]
    assert isinstance(stmt3.target, ArrayLValueNode), "Target debe ser ArrayLValueNode"
    
    return True


def test_complex_expressions():
    """Test 9: Expresiones complejas"""
    code = """
Complex()
begin
    x ← (5 + 3) * 2
    y ← 2 ^ 3 ^ 2
    z ← -5 + 3
end
    """
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    statements = ast.procedures[0].body.statements
    
    # Primera: (5 + 3) * 2
    stmt1 = statements[0]
    assert isinstance(stmt1.value, BinaryOpNode), "Debe ser BinaryOpNode"
    assert stmt1.value.op == "*", "Debe ser multiplicación"
    
    # Segunda: 2 ^ 3 ^ 2 (asociativa a la derecha)
    stmt2 = statements[1]
    assert isinstance(stmt2.value, BinaryOpNode), "Debe ser BinaryOpNode"
    assert stmt2.value.op == "^", "Debe ser potencia"
    
    # Tercera: -5 + 3 (unario)
    stmt3 = statements[2]
    assert isinstance(stmt3.value, BinaryOpNode), "Debe ser BinaryOpNode"
    assert stmt3.value.op == "+", "Debe ser suma"
    
    return True


def test_call_statement():
    """Test 10: Llamadas a procedimientos"""
    code = """
Main()
begin
    call Sort(A, n)
    x ← call GetValue()
end
    """
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    statements = ast.procedures[0].body.statements
    
    # Primera: call como statement
    stmt1 = statements[0]
    assert isinstance(stmt1, CallStatementNode), "Debe ser CallStatementNode"
    assert stmt1.name == "Sort", "Debe llamar a Sort"
    assert len(stmt1.arguments) == 2, "Debe tener 2 argumentos"
    
    # Segunda: call como expresión
    stmt2 = statements[1]
    assert isinstance(stmt2, AssignmentNode), "Debe ser AssignmentNode"
    assert isinstance(stmt2.value, FunctionCallNode), "Debe ser FunctionCallNode"
    assert stmt2.value.name == "GetValue", "Debe llamar a GetValue"
    
    return True


# ============================================================================
# RUNNER DE TESTS
# ============================================================================

def run_all_tests():
    """Ejecuta todos los tests"""
    
    tests = [
        (test_simple_assignment, "Test 1: Asignación Simple"),
        (test_for_loop, "Test 2: Ciclo FOR"),
        (test_binary_operations, "Test 3: Operaciones Binarias"),
        (test_nested_for, "Test 4: FOR Anidado"),
        (test_while_loop, "Test 5: Ciclo WHILE"),
        (test_if_statement, "Test 6: IF-THEN-ELSE"),
        (test_recursion, "Test 7: Recursión (Factorial)"),
        (test_array_access, "Test 8: Acceso a Arrays"),
        (test_complex_expressions, "Test 9: Expresiones Complejas"),
        (test_call_statement, "Test 10: Llamadas a Procedimientos"),
    ]
    
    print("="*70)
    print("TEST COMPLETO DEL SISTEMA AST")
    print("="*70)
    print()
    
    results = []
    
    for test_func, test_name in tests:
        print(f"{'='*70}")
        print(f"🧪 {test_name}")
        print(f"{'='*70}")
        
        try:
            success = test_func()
            print(f"✓ PASS")
            results.append((test_name, True, None))
        
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
            results.append((test_name, False, str(e)))
        
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append((test_name, False, f"Error inesperado: {e}"))
        
        print()
    
    # Resumen
    print("="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {name}")
        if error:
            print(f"       {error}")
    
    print("="*70)
    percentage = (passed / total * 100) if total > 0 else 0
    print(f"Resultado: {passed}/{total} tests pasados ({percentage:.1f}%)")
    print("="*70)
    
    return passed == total


# ============================================================================
# MODO VISUAL: Mostrar AST de un código
# ============================================================================

def visualize_ast(code: str):
    """Muestra el AST de un código de forma legible"""
    
    parser = load_parser()
    tree = parser.parse(code)
    ast = transform_to_ast(tree)
    
    def print_node(node, indent=0):
        """Imprime un nodo recursivamente"""
        prefix = "  " * indent
        
        if isinstance(node, ProgramNode):
            print(f"{prefix}Program")
            for proc in node.procedures:
                print_node(proc, indent + 1)
        
        elif isinstance(node, ProcedureNode):
            print(f"{prefix}Procedure: {node.name}")
            print(f"{prefix}  Params: {[p.name for p in node.parameters]}")
            print(f"{prefix}  Body:")
            print_node(node.body, indent + 2)
        
        elif isinstance(node, ForNode):
            print(f"{prefix}FOR {node.variable} = {node.start.value if isinstance(node.start, NumberNode) else '?'} to {node.end.name if isinstance(node.end, IdentifierNode) else '?'}")
            print_node(node.body, indent + 1)
        
        elif isinstance(node, AssignmentNode):
            target_name = node.target.name if hasattr(node.target, 'name') else str(node.target)
            print(f"{prefix}Assignment: {target_name} ← ...")
        
        elif hasattr(node, 'statements'):
            for stmt in node.statements:
                print_node(stmt, indent)
    
    print_node(ast)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test del sistema AST")
    parser.add_argument('--visualize', '-v', action='store_true',
                       help="Modo visualización de AST")
    parser.add_argument('--code', '-c', type=str,
                       help="Código a visualizar")
    
    args = parser.parse_args()
    
    if args.visualize and args.code:
        print("="*70)
        print("VISUALIZACIÓN DE AST")
        print("="*70)
        print("\nCódigo:")
        print(args.code)
        print("\nAST:")
        visualize_ast(args.code)
    
    else:
        # Ejecutar tests
        success = run_all_tests()
        sys.exit(0 if success else 1)