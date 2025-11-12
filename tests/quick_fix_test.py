"""
Script de Verificación Rápida - Testear Fixes
=============================================

Ejecutar: python quick_fix_test.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.parser.parser import parse, ParseError
from src.analyzer.visitor import CountingVisitor
from lark import Token


# ============================================================================
# TEST 1: Asignación Simple (el que fallaba)
# ============================================================================

def test_simple_assignment():
    """Test que fallaba: Token object has no attribute 'name'"""
    code = """
Simple()
begin
    x ← 5
    y ← x + 3
end
    """
    
    print("="*70)
    print("TEST 1: Asignación Simple")
    print("="*70)
    
    try:
        ast = parse(code)
        
        # Verificar que no hay Tokens residuales
        def check_tokens(node, path=""):
            if isinstance(node, Token):
                print(f"  ✗ Token encontrado en: {path}")
                return False
            
            if hasattr(node, '__dict__'):
                for attr, value in node.__dict__.items():
                    new_path = f"{path}.{attr}" if path else attr
                    
                    if isinstance(value, Token):
                        print(f"  ✗ Token en: {new_path}")
                        return False
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if not check_tokens(item, f"{new_path}[{i}]"):
                                return False
                    elif hasattr(value, '__dict__'):
                        if not check_tokens(value, new_path):
                            return False
            return True
        
        if check_tokens(ast):
            print("  ✓ Sin Tokens residuales")
        
        # Probar CountingVisitor
        visitor = CountingVisitor()
        counts = visitor.visit_program(ast)
        
        print(f"  ✓ CountingVisitor funcionó")
        print(f"  ✓ Asignaciones: {counts['Simple'].assignments}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 2: Clases y Objetos (el que fallaba)
# ============================================================================

def test_classes_and_objects():
    """Test que fallaba: Error con acceso a objetos"""
    code = """
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
    
    print("\n" + "="*70)
    print("TEST 2: Clases y Objetos")
    print("="*70)
    
    try:
        ast = parse(code)
        
        print(f"  ✓ Parseado exitoso")
        print(f"  ✓ Clases: {len(ast.classes)}")
        print(f"  ✓ Procedimientos: {len(ast.procedures)}")
        
        # Verificar clase
        if ast.classes:
            clase = ast.classes[0]
            print(f"  ✓ Clase: {clase.name}")
            print(f"  ✓ Atributos: {clase.attributes}")
        
        # Probar visitor
        visitor = CountingVisitor()
        counts = visitor.visit_program(ast)
        
        print(f"  ✓ CountingVisitor funcionó")
        
        return True
    
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 3: Bubble Sort (verificar que sigue funcionando)
# ============================================================================

def test_bubble_sort():
    """Verificar que Bubble Sort sigue funcionando"""
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
    print("TEST 3: Bubble Sort")
    print("="*70)
    
    try:
        ast = parse(code)
        
        visitor = CountingVisitor()
        counts = visitor.visit_program(ast)
        
        proc_count = counts['BubbleSort']
        
        print(f"  ✓ Parseado exitoso")
        print(f"  ✓ Asignaciones: {proc_count.assignments}")
        print(f"  ✓ Comparaciones: {proc_count.comparisons}")
        print(f"  ✓ Accesos a arrays: {proc_count.array_accesses}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ejecutar todos los tests de verificación"""
    
    print("="*70)
    print("VERIFICACIÓN RÁPIDA DE FIXES")
    print("="*70)
    print()
    
    tests = [
        ("Asignación Simple", test_simple_assignment),
        ("Clases y Objetos", test_classes_and_objects),
        ("Bubble Sort", test_bubble_sort),
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
        print("\n🎉 ¡Todos los tests pasaron!")
        print("✅ Ahora puedes ejecutar:")
        print("   • python tests/test_parser.py")
        print("   • python -m src.analyzer.visitor")
        print("   • python demo.py")
    else:
        print("\n⚠️  Algunos tests fallaron. Revisar errores arriba.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)