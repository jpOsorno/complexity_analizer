"""
Tests del Analizador de Recursión - VERSIÓN CORREGIDA
=====================================================

Ejecutar: python tests/test_recursion.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.parser import parse
from analyzer.recursion_analyzer import analyze_recursion, to_recurrence


# ============================================================================
# TEST 1: Factorial
# ============================================================================

def test_factorial():
    """Test: Factorial - recursión lineal"""
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
    
    print("="*70)
    print("TEST 1: Factorial (Recursión Lineal)")
    print("="*70)
    
    try:
        ast = parse(code)
        proc = ast.procedures[0]
        result = analyze_recursion(proc)
        
        assert result.is_recursive, "Debe detectar que es recursivo"
        assert result.recurrence_equation is not None, "Debe tener ecuación"
        
        eq = result.recurrence_equation
        
        print(f"✓ Detectado como recursivo")
        print(f"✓ Ecuación: {eq.equation_str}")
        print(f"✓ Tipo: {eq.recursion_type}")
        
        # Test de to_recurrence()
        equation_str = to_recurrence(code)
        print(f"✓ to_recurrence(): {equation_str}")
        assert equation_str is not None, "to_recurrence() debe retornar ecuación"
        
        # Verificar que contiene elementos esperados
        assert "T(n)" in equation_str, "Debe tener T(n)"
        assert "T(n-1)" in equation_str or "T(n-1)" in eq.equation_str, "Debe tener T(n-1)"
        assert "O(1)" in equation_str, "Debe tener O(1)"
        
        print("\n✅ TEST PASS: Factorial")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAIL: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 2: Fibonacci
# ============================================================================

def test_fibonacci():
    """Test: Fibonacci - recursión binaria"""
    code = """
Fibonacci(n)
begin
    if (n ≤ 1) then
    begin
        return n
    end
    else
    begin
        return call Fibonacci(n-1) + call Fibonacci(n-2)
    end
end
    """
    
    print("\n" + "="*70)
    print("TEST 2: Fibonacci (Recursión Binaria)")
    print("="*70)
    
    try:
        ast = parse(code)
        proc = ast.procedures[0]
        result = analyze_recursion(proc)
        
        assert result.is_recursive, "Debe detectar que es recursivo"
        
        eq = result.recurrence_equation
        
        print(f"✓ Detectado como recursivo")
        print(f"✓ Ecuación: {eq.equation_str}")
        print(f"✓ Tipo: {eq.recursion_type}")
        
        # Test de to_recurrence()
        equation_str = to_recurrence(code)
        print(f"✓ to_recurrence(): {equation_str}")
        
        assert eq.recursion_type == "binary", "Debe ser recursión binaria"
        assert len(eq.recursive_calls) == 2, "Debe tener 2 llamadas recursivas"
        assert "T(n-1)" in equation_str and "T(n-2)" in equation_str, "Debe tener ambos términos"
        
        print("\n✅ TEST PASS: Fibonacci")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAIL: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 3: Binary Search (SIMPLIFICADO para evitar ambigüedad)
# ============================================================================

def test_binary_search_simple():
    """Test: Binary Search simplificado"""
    code = """
BinarySearch(A[], left, right, x)
begin
    if (left > right) then
    begin
        return -1
    end
    
    mid ← floor((left + right) / 2)
    
    if (A[mid] < x) then
    begin
        return call BinarySearch(A, mid+1, right, x)
    end
    
    return mid
end
    """
    
    print("\n" + "="*70)
    print("TEST 3: Binary Search Simplificado")
    print("="*70)
    
    try:
        ast = parse(code)
        proc = ast.procedures[0]
        result = analyze_recursion(proc)
        
        assert result.is_recursive, "Debe detectar que es recursivo"
        
        eq = result.recurrence_equation
        
        print(f"✓ Detectado como recursivo")
        print(f"✓ Ecuación: {eq.equation_str}")
        print(f"✓ Tipo: {eq.recursion_type}")
        
        equation_str = to_recurrence(code)
        print(f"✓ to_recurrence(): {equation_str}")
        
        assert len(eq.recursive_calls) >= 1, "Debe tener llamadas recursivas"
        
        print("\n✅ TEST PASS: Binary Search")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAIL: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 4: Merge Sort
# ============================================================================

def test_merge_sort():
    """Test: Merge Sort - recursión múltiple"""
    code = """
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
    
    print("\n" + "="*70)
    print("TEST 4: Merge Sort (Múltiple)")
    print("="*70)
    
    try:
        ast = parse(code)
        proc = ast.procedures[0]
        result = analyze_recursion(proc)
        
        assert result.is_recursive, "Debe detectar que es recursivo"
        
        eq = result.recurrence_equation
        
        print(f"✓ Detectado como recursivo")
        print(f"✓ Ecuación: {eq.equation_str}")
        print(f"✓ Tipo: {eq.recursion_type}")
        
        equation_str = to_recurrence(code)
        print(f"✓ to_recurrence(): {equation_str}")
        
        assert len(eq.recursive_calls) == 2, "Debe tener 2 llamadas recursivas"
        assert eq.recursion_type in ["binary", "multiple"], "Debe ser binaria/múltiple"
        
        print("\n✅ TEST PASS: Merge Sort")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAIL: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 5: Bubble Sort (No Recursivo)
# ============================================================================

def test_non_recursive():
    """Test: Bubble Sort - no recursivo"""
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
    
    print("\n" + "="*70)
    print("TEST 5: Bubble Sort (No Recursivo)")
    print("="*70)
    
    try:
        ast = parse(code)
        proc = ast.procedures[0]
        result = analyze_recursion(proc)
        
        assert not result.is_recursive, "No debe detectar recursión"
        assert result.recurrence_equation is None, "No debe tener ecuación"
        
        # Test de to_recurrence()
        equation_str = to_recurrence(code)
        assert equation_str is None, "to_recurrence() debe retornar None"
        
        print(f"✓ Correctamente identificado como NO recursivo")
        print(f"✓ to_recurrence() retorna None")
        
        print("\n✅ TEST PASS: Bubble Sort")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAIL: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST EXTRA: API to_recurrence()
# ============================================================================

def test_to_recurrence_api():
    """Test: Función to_recurrence() directa"""
    print("\n" + "="*70)
    print("TEST 6: API to_recurrence()")
    print("="*70)
    
    # Test con código simple
    factorial_code = """
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
    
    try:
        eq = to_recurrence(factorial_code)
        
        assert eq is not None, "Debe retornar ecuación"
        assert "T(n)" in eq, "Debe contener T(n)"
        assert "O(1)" in eq, "Debe contener O(1)"
        
        print(f"✓ Factorial: {eq}")
        
        # Test con código no recursivo
        bubble_code = "BubbleSort(A[], n)\nbegin\n  for i ← 1 to n do\n  begin\n    x ← i\n  end\nend"
        eq2 = to_recurrence(bubble_code)
        
        assert eq2 is None, "No recursivo debe retornar None"
        print(f"✓ Bubble Sort: None (correcto)")
        
        print("\n✅ TEST PASS: to_recurrence() API")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAIL: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ejecutar todos los tests"""
    
    print("="*70)
    print("TESTS DEL ANALIZADOR DE RECURSIÓN")
    print("="*70)
    print()
    
    tests = [
        ("Factorial", test_factorial),
        ("Fibonacci", test_fibonacci),
        ("Binary Search Simple", test_binary_search_simple),
        ("Merge Sort", test_merge_sort),
        ("Bubble Sort (no recursivo)", test_non_recursive),
        ("API to_recurrence()", test_to_recurrence_api),
    ]
    
    results = []
    
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {name}")
    
    print("="*70)
    print(f"Resultado: {passed}/{total} ({passed/total*100:.0f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ¡Todos los tests de recursión pasaron!")
        print("\n📚 Funcionalidad Completada:")
        print("   ✓ Detección de recursión")
        print("   ✓ Generación de ecuaciones de recurrencia")
        print("   ✓ Clasificación (lineal, binaria, divide-and-conquer)")
        print("   ✓ API to_recurrence() para obtener solo la ecuación")
        print("\n💡 Próximo Paso:")
        print("   → Implementar resolutor de ecuaciones de recurrencia")
        print("   → Master Theorem + Método de sustitución + Árboles de recursión")
    else:
        print("\n⚠️  Algunos tests fallaron")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)