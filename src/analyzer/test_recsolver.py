"""
Tests del Resolutor de Recurrencias
===================================

Prueba todas las técnicas implementadas.
Ejecutar: python tests/test_solver.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analyzer.recurrence_solver import solve_recurrence, MasterTheorem, RecursionTree


# ============================================================================
# TEST 1: Teorema Maestro - Caso 1
# ============================================================================

def test_master_theorem_case1():
    """Test: Binary Search - Caso 1 del Teorema Maestro"""
    print("="*70)
    print("TEST 1: Binary Search (Master Theorem Caso 1)")
    print("="*70)
    
    equation = "T(n) = T(n/2) + O(1)"
    
    try:
        solution = solve_recurrence(equation)
        
        print(f"\nEcuación: {equation}")
        print(f"Método: {solution.method_used}")
        print(f"Θ(n): {solution.big_theta}")
        
        assert "master" in solution.method_used.lower(), "Debe usar Teorema Maestro"
        assert "log" in solution.big_theta.lower(), "Binary Search es O(log n)"
        
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
# TEST 2: Teorema Maestro - Caso 2
# ============================================================================

def test_master_theorem_case2():
    """Test: Merge Sort - Caso 2 del Teorema Maestro"""
    print("\n" + "="*70)
    print("TEST 2: Merge Sort (Master Theorem Caso 2)")
    print("="*70)
    
    equation = "T(n) = 2T(n/2) + O(n)"
    
    try:
        solution = solve_recurrence(equation)
        
        print(f"\nEcuación: {equation}")
        print(f"Método: {solution.method_used}")
        print(f"Θ(n): {solution.big_theta}")
        
        assert "master" in solution.method_used.lower(), "Debe usar Teorema Maestro"
        assert "log" in solution.big_theta.lower() and "n" in solution.big_theta, "Merge Sort es O(n log n)"
        
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
# TEST 3: Teorema Maestro - Caso 3
# ============================================================================

def test_master_theorem_case3():
    """Test: Strassen - Caso 3 del Teorema Maestro"""
    print("\n" + "="*70)
    print("TEST 3: Strassen (Master Theorem Caso 3)")
    print("="*70)
    
    equation = "T(n) = 7T(n/2) + O(n^2)"
    
    try:
        solution = solve_recurrence(equation)
        
        print(f"\nEcuación: {equation}")
        print(f"Método: {solution.method_used}")
        print(f"Θ(n): {solution.big_theta}")
        
        assert "master" in solution.method_used.lower(), "Debe usar Teorema Maestro"
        # Strassen es O(n^2.81) aproximadamente
        
        print("\n✅ TEST PASS: Strassen")
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
# TEST 4: Árbol de Recursión - Lineal
# ============================================================================

def test_recursion_tree_linear():
    """Test: Factorial - Árbol de Recursión Lineal"""
    print("\n" + "="*70)
    print("TEST 4: Factorial (Árbol de Recursión)")
    print("="*70)
    
    equation = "T(n) = T(n-1) + O(1)"
    
    try:
        solution = solve_recurrence(equation)
        
        print(f"\nEcuación: {equation}")
        print(f"Método: {solution.method_used}")
        print(f"Θ(n): {solution.big_theta}")
        
        assert "Θ(n)" in solution.big_theta, "Factorial es O(n)"
        assert solution.complexity_class == "linear", "Debe ser lineal"
        
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
# TEST 5: Árbol de Recursión - Fibonacci
# ============================================================================

def test_recursion_tree_fibonacci():
    """Test: Fibonacci - Árbol de Recursión Exponencial"""
    print("\n" + "="*70)
    print("TEST 5: Fibonacci (Árbol de Recursión)")
    print("="*70)
    
    equation = "T(n) = T(n-1) + T(n-2) + O(1)"
    
    try:
        solution = solve_recurrence(equation)
        
        print(f"\nEcuación: {equation}")
        print(f"Método: {solution.method_used}")
        print(f"Θ(n): {solution.big_theta}")
        
        assert "exponential" in solution.complexity_class, "Fibonacci es exponencial"
        assert "2^n" in solution.big_o or "φ" in solution.big_theta, "Debe mencionar 2^n o φ^n"
        
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
# TEST 6: Integración con Recursion Analyzer
# ============================================================================

def test_integration_with_recursion_analyzer():
    """Test: Integración completa desde código hasta solución"""
    print("\n" + "="*70)
    print("TEST 6: Integración Completa")
    print("="*70)
    
    try:
        from src.parser.parser import parse
        from src.analyzer.recursion_analyzer import analyze_recursion
        
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
        
        # Paso 1: Parsear
        ast = parse(code)
        proc = ast.procedures[0]
        
        # Paso 2: Analizar recursión
        rec_result = analyze_recursion(proc)
        equation = rec_result.to_recurrence()
        
        print(f"\nCódigo parseado: Factorial")
        print(f"Ecuación extraída: {equation}")
        
        # Paso 3: Resolver recurrencia
        solution = solve_recurrence(equation)
        
        print(f"Método: {solution.method_used}")
        print(f"Complejidad: {solution.big_theta}")
        
        assert equation is not None, "Debe extraer ecuación"
        assert "Θ(n)" in solution.big_theta, "Factorial es O(n)"
        
        print("\n✅ TEST PASS: Integración Completa")
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
# TEST 7: Notaciones O, Ω, Θ
# ============================================================================

def test_notations():
    """Test: Verificar notaciones O, Ω, Θ correctas"""
    print("\n" + "="*70)
    print("TEST 7: Notaciones O, Ω, Θ")
    print("="*70)
    
    equation = "T(n) = 2T(n/2) + O(n)"
    
    try:
        solution = solve_recurrence(equation)
        
        print(f"\nEcuación: {equation}")
        print(f"O(n): {solution.big_o}")
        print(f"Ω(n): {solution.big_omega}")
        print(f"Θ(n): {solution.big_theta}")
        print(f"Tight bound: {solution.is_tight}")
        
        assert solution.big_o is not None, "Debe tener Big-O"
        assert solution.big_omega is not None, "Debe tener Big-Omega"
        assert solution.big_theta is not None, "Debe tener Big-Theta"
        assert solution.is_tight, "Merge Sort tiene tight bound"
        
        print("\n✅ TEST PASS: Notaciones")
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
    print("TESTS DEL RESOLUTOR DE RECURRENCIAS")
    print("="*70)
    print()
    
    tests = [
        ("Binary Search (Master Caso 1)", test_master_theorem_case1),
        ("Merge Sort (Master Caso 2)", test_master_theorem_case2),
        ("Strassen (Master Caso 3)", test_master_theorem_case3),
        ("Factorial (Árbol Lineal)", test_recursion_tree_linear),
        ("Fibonacci (Árbol Exponencial)", test_recursion_tree_fibonacci),
        ("Integración Completa", test_integration_with_recursion_analyzer),
        ("Notaciones O, Ω, Θ", test_notations),
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
        print("\n🎉 ¡Todos los tests del resolutor pasaron!")
        print("\n✅ Técnicas Implementadas:")
        print("   ✓ Teorema Maestro (3 casos)")
        print("   ✓ Árbol de Recursión")
        print("   ✓ Método de Sustitución con SymPy")
        print("   ✓ Notaciones O, Ω, Θ")
        print("   ✓ Simplificación simbólica")
        print("\n💡 Próximos Pasos:")
        print("   → Integrar con análisis de ciclos iterativos")
        print("   → Comparación con LLM (Claude/GPT)")
        print("   → Diagramas de seguimiento")
    else:
        print("\n⚠️  Algunos tests fallaron")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)