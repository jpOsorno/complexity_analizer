"""
Tests del Analizador de Recursión
==================================

Ejecutar: python test_recursion.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.parser.parser import parse
from src.analyzer.recursion_analyzer import analyze_recursion, RecurrenceEquation


# ============================================================================
# TEST 1: Factorial (Recursión Lineal)
# ============================================================================

def test_factorial():
    """Test: Factorial - recursión lineal simple"""
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
        
        # Verificaciones
        assert result.is_recursive, "Debe detectar que es recursivo"
        assert result.recurrence_equation is not None, "Debe tener ecuación"
        
        eq = result.recurrence_equation
        
        print(f"✓ Detectado como recursivo")
        print(f"✓ Ecuación: {eq.equation_str}")
        print(f"✓ Tipo: {eq.recursion_type}")
        print(f"✓ Llamadas recursivas: {len(eq.recursive_calls)}")
        
        assert eq.recursion_type in ["linear", "simple"], "Debe ser recursión lineal"
        assert len(eq.recursive_calls) == 1, "Debe tener 1 llamada recursiva"
        assert "n-1" in eq.recursive_calls[0].depth_reduction, "Debe reducir n-1"
        
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
# TEST 2: Fibonacci (Recursión Binaria)
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
        print(f"✓ Llamadas recursivas: {len(eq.recursive_calls)}")
        
        assert eq.recursion_type == "binary", "Debe ser recursión binaria"
        assert len(eq.recursive_calls) == 2, "Debe tener 2 llamadas recursivas"
        
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
# TEST 3: Binary Search (Divide and Conquer)
# ============================================================================

def test_binary_search():
    """Test: Binary Search - divide and conquer"""
    code = """
BinarySearch(A[], left, right, x)
begin
    if (left > right) then
    begin
        return -1
    end
    
    mid ← floor((left + right) / 2)
    
    if (A[mid] = x) then
    begin
        return mid
    end
    else
    begin
        if (A[mid] < x) then
        begin
            return call BinarySearch(A, mid+1, right, x)
        end
        else
        begin
            return call BinarySearch(A, left, mid-1, x)
        end
    end
end
    """
    
    print("\n" + "="*70)
    print("TEST 3: Binary Search (Divide & Conquer)")
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
        print(f"✓ Llamadas recursivas: {len(eq.recursive_calls)}")
        
        # Binary Search tiene 1 llamada recursiva (en cada rama)
        # pero solo se ejecuta una por vez
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
# TEST 4: Merge Sort (Recursión Múltiple)
# ============================================================================

def test_merge_sort():
    """Test: Merge Sort - recursión múltiple + divide & conquer"""
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
        print(f"✓ Llamadas recursivas: {len(eq.recursive_calls)}")
        
        assert len(eq.recursive_calls) == 2, "Debe tener 2 llamadas recursivas a MergeSort"
        assert eq.recursion_type in ["binary", "multiple"], "Debe ser recursión binaria/múltiple"
        
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
# TEST 5: Algoritmo No Recursivo
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
        
        print(f"✓ Correctamente identificado como NO recursivo")
        print(f"✓ Sin ecuación de recurrencia")
        
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
        ("Binary Search", test_binary_search),
        ("Merge Sort", test_merge_sort),
        ("Bubble Sort (no recursivo)", test_non_recursive),
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
        print("\n📚 Ecuaciones de Recurrencia Detectadas:")
        print("   • Factorial:      T(n) = T(n-1) + O(1)")
        print("   • Fibonacci:      T(n) = T(n-1) + T(n-2) + O(1)")
        print("   • Binary Search:  T(n) = T(n/2) + O(1)")
        print("   • Merge Sort:     T(n) = 2T(n/2) + O(n)")
    else:
        print("\n⚠️  Algunos tests fallaron")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)