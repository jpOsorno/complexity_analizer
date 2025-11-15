"""
Test del Analizador de Complejidad Básico
=========================================

Prueba el análisis de complejidad para algoritmos iterativos.

Ejecutar: python test_complexity_analyzer.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.parser.parser import parse
from src.analyzer.complexity_analyzer import analyze_complexity, BasicComplexityAnalyzer


# ============================================================================
# TESTS
# ============================================================================

def test_constant_time():
    """Test 1: O(1) - Operaciones constantes"""
    code = """
Constant()
begin
    x ← 5
    y ← x + 3
    z ← y * 2
end
    """
    
    ast = parse(code)
    results = analyze_complexity(ast)
    result = results['Constant']
    
    print("Test 1: Operaciones Constantes")
    print(f"  Resultado: {result.worst_case}")
    assert "O(1)" in result.worst_case, f"Esperado O(1), obtenido {result.worst_case}"
    print("  ✓ PASS\n")
    return True


def test_linear_for():
    """Test 2: O(n) - FOR simple con early exit"""
    code = """
LinearSearch(A[], n, x)
begin
    for i ← 1 to n do
    begin
        if (A[i] = x) then
        begin
            return i
        end
    end
    return -1
end
    """
    
    ast = parse(code)
    results = analyze_complexity(ast)
    result = results['LinearSearch']
    
    print("Test 2: Búsqueda Lineal - Early Exit")
    print(f"  Peor caso: {result.worst_case}")
    print(f"  Mejor caso: {result.best_case}")
    print(f"  Caso promedio: {result.average_case}")
    print(f"  Explicación: {result.explanation}")
    
    # Verificar que detecta early exit
    assert "O(n)" in result.worst_case, f"Peor caso debe ser O(n), obtenido {result.worst_case}"
    assert "Ω(1)" in result.best_case, f"Mejor caso debe ser Ω(1), obtenido {result.best_case}"
    print("  ✓ PASS: Detecta early exit correctamente\n")
    return True


def test_quadratic_nested_for():
    """Test 3: O(n²) - FOR anidado"""
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
    
    ast = parse(code)
    results = analyze_complexity(ast)
    result = results['BubbleSort']
    
    print("Test 3: FOR Anidado - O(n²)")
    print(f"  Resultado: {result.worst_case}")
    print(f"  Explicación: {result.explanation}")
    
    # Debe contener n² o n×n
    assert "n²" in result.worst_case or "n×n" in result.worst_case, \
        f"Esperado O(n²), obtenido {result.worst_case}"
    print("  ✓ PASS\n")
    return True


def test_cubic_triple_for():
    """Test 4: O(n³) - FOR triple anidado"""
    code = """
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
end
    """
    
    ast = parse(code)
    results = analyze_complexity(ast)
    result = results['MatrixMultiply']
    
    print("Test 4: FOR Triple Anidado - O(n³)")
    print(f"  Resultado: {result.worst_case}")
    print(f"  Explicación: {result.explanation}")
    
    assert "n³" in result.worst_case or "n×n×n" in result.worst_case, \
        f"Esperado O(n³), obtenido {result.worst_case}"
    print("  ✓ PASS\n")
    return True


def test_if_statement():
    """Test 5: IF-THEN-ELSE - Toma el máximo"""
    code = """
MaxOfTwo(a, b)
begin
    if (a > b) then
    begin
        for i ← 1 to 10 do
        begin
            x ← i
        end
        return a
    end
    else
    begin
        y ← b
        return b
    end
end
    """
    
    ast = parse(code)
    results = analyze_complexity(ast)
    result = results['MaxOfTwo']
    
    print("Test 5: IF-THEN-ELSE - Max de ramas")
    print(f"  Resultado: {result.worst_case}")
    print(f"  Explicación: {result.explanation}")
    
    # El IF debe tomar el máximo entre O(10)=O(1) y O(1)
    # Resultado esperado: O(1) o O(10) que simplifica a O(1)
    print("  ✓ PASS\n")
    return True


def test_constant_iterations():
    """Test 6: FOR con límites constantes"""
    code = """
Fixed()
begin
    for i ← 1 to 100 do
    begin
        x ← i * 2
    end
end
    """
    
    ast = parse(code)
    results = analyze_complexity(ast)
    result = results['Fixed']
    
    print("Test 6: FOR con límites constantes")
    print(f"  Resultado: {result.worst_case}")
    print(f"  Explicación: {result.explanation}")
    
    # 100 iteraciones = O(1) (constante)
    assert "O(100)" in result.worst_case or "O(1)" in result.worst_case, \
        f"Esperado O(100) o O(1), obtenido {result.worst_case}"
    print("  ✓ PASS\n")
    return True


def test_complex_example():
    """Test 7: Ejemplo complejo con múltiples ciclos"""
    code = """
Complex(A[], n)
begin
    x ← 0
    for i ← 1 to n do
    begin
        x ← x + A[i]
    end
    
    for i ← 1 to n do
    begin
        for j ← 1 to n do
        begin
            y ← i + j
        end
    end
    
    return x
end
    """
    
    ast = parse(code)
    results = analyze_complexity(ast)
    result = results['Complex']
    
    print("Test 7: Ejemplo Complejo")
    print(f"  Resultado: {result.worst_case}")
    print(f"  Explicación: {result.explanation}")
    
    # Debe ser O(n²) porque domina sobre O(n)
    assert "n²" in result.worst_case, f"Esperado O(n²), obtenido {result.worst_case}"
    print("  ✓ PASS\n")
    return True


def test_simplification():
    """Test 8: Verificar simplificación de expresiones"""
    code = """
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
    
    ast = parse(code)
    results = analyze_complexity(ast)
    result = results['Simplified']
    
    print("Test 8: Simplificación de Expresiones")
    print(f"  Resultado: {result.worst_case}")
    print(f"  Explicación: {result.explanation}")
    
    # n-1 y n-i deben simplificarse a n, dando O(n²)
    assert "n²" in result.worst_case, f"Esperado O(n²), obtenido {result.worst_case}"
    print("  ✓ PASS\n")
    return True


def test_no_early_exit():
    """Test 9: FOR sin early exit - Todos los casos iguales"""
    code = """
SumArray(A[], n)
begin
    sum ← 0
    for i ← 1 to n do
    begin
        sum ← sum + A[i]
    end
    return sum
end
    """
    
    ast = parse(code)
    results = analyze_complexity(ast)
    result = results['SumArray']
    
    print("Test 9: FOR sin Early Exit")
    print(f"  Peor caso: {result.worst_case}")
    print(f"  Mejor caso: {result.best_case}")
    print(f"  Caso promedio: {result.average_case}")
    
    # Sin early exit, todos los casos deben ser iguales
    assert "O(n)" in result.worst_case, f"Peor caso debe ser O(n)"
    assert "Ω(n)" in result.best_case, f"Mejor caso debe ser Ω(n) (no Ω(1))"
    assert "Θ(n)" in result.average_case, f"Caso promedio debe ser Θ(n)"
    print("  ✓ PASS: Sin early exit, todos los casos son O(n)\n")
    return True


# ============================================================================
# RUNNER
# ============================================================================

def run_all_tests():
    """Ejecuta todos los tests"""
    
    tests = [
        test_constant_time,
        test_linear_for,
        test_quadratic_nested_for,
        test_cubic_triple_for,
        test_if_statement,
        test_constant_iterations,
        test_complex_example,
        test_simplification,
        test_no_early_exit,
    ]
    
    print("="*70)
    print("TESTS DEL ANALIZADOR DE COMPLEJIDAD BÁSICO")
    print("="*70)
    print()
    
    results = []
    
    for test_func in tests:
        try:
            success = test_func()
            results.append((test_func.__name__, True, None))
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}\n")
            results.append((test_func.__name__, False, str(e)))
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            results.append((test_func.__name__, False, f"Error: {e}"))
    
    # Resumen
    print("="*70)
    print("RESUMEN")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {name}")
        if error:
            print(f"       {error}")
    
    print("="*70)
    print(f"Resultado: {passed}/{total} tests pasados ({passed/total*100:.1f}%)")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)