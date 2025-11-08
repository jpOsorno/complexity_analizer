"""
Script de prueba básico para la gramática Lark
Ejecutar: python test_grammar.py
"""

from lark import Lark, Tree
from lark.exceptions import LarkError
import sys

# ============================================================================
# CARGAR LA GRAMÁTICA
# ============================================================================
def load_grammar():
    """Carga la gramática desde el archivo grammar.lark"""
    try:
        # Obtener el directorio donde está este script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        grammar_path = os.path.join(script_dir, 'grammar.lark')
        
        with open(grammar_path, 'r', encoding='utf-8') as f:
            grammar = f.read()
        
        # Crear el parser con configuración óptima
        parser = Lark(
            grammar,
            parser='earley',        # Más rápido que earley
            start='program',      # Símbolo inicial
            propagate_positions=True,  # Para debugging
            maybe_placeholders=False   # Errores más claros
        )
        
        print("✓ Gramática cargada exitosamente")
        print(f"   Ruta: {grammar_path}")
        return parser
    
    except FileNotFoundError:
        print(f"✗ Error: No se encontró el archivo 'grammar.lark'")
        print(f"   Buscado en: {grammar_path}")
        print(f"   Directorio actual: {os.getcwd()}")
        print(f"   Directorio del script: {script_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error al cargar la gramática: {e}")
        sys.exit(1)

# ============================================================================
# FUNCIÓN DE PRUEBA
# ============================================================================
def test_parse(parser, code, test_name="Test"):
    """
    Intenta parsear un código y muestra el resultado
    
    Args:
        parser: Parser de Lark
        code: Código pseudocódigo a parsear
        test_name: Nombre descriptivo del test
    """
    print(f"\n{'='*70}")
    print(f"🧪 {test_name}")
    print(f"{'='*70}")
    print("Código:")
    print("-" * 70)
    print(code)
    print("-" * 70)
    
    try:
        tree = parser.parse(code)
        print("✓ PARSEO EXITOSO")
        print("\nÁrbol de parseo:")
        print(tree.pretty())
        return True
    
    except LarkError as e:
        print("✗ ERROR DE PARSEO")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {e}")
        
        # Intentar dar más contexto del error
        if hasattr(e, 'line'):
            print(f"Línea: {e.line}")
        if hasattr(e, 'column'):
            print(f"Columna: {e.column}")
        
        return False
    
    except Exception as e:
        print(f"✗ ERROR INESPERADO: {e}")
        return False

# ============================================================================
# CASOS DE PRUEBA
# ============================================================================

# TEST 1: Asignación simple
TEST_1_ASSIGNMENT = """
Simple()
begin
    x ← 5
end
"""

# TEST 2: Ciclo FOR básico
TEST_2_FOR = """
ForBasico()
begin
    for i ← 1 to 10 do
    begin
        x ← i
    end
end
"""

# TEST 3: Ciclo WHILE
TEST_3_WHILE = """
WhileBasico()
begin
    i ← 1
    while (i < 10) do
    begin
        i ← i + 1
    end
end
"""

# TEST 4: IF-THEN-ELSE
TEST_4_IF = """
IfBasico()
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

# TEST 5: Arrays
TEST_5_ARRAY = """
ArrayBasico(A[], n)
begin
    for i ← 1 to n do
    begin
        A[i] ← 0
    end
end
"""

# TEST 6: Recursión (Factorial)
TEST_6_RECURSION = """
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

# TEST 7: Bubble Sort
TEST_7_BUBBLE_SORT = """
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

# TEST 8: Con clase y objetos
TEST_8_OBJECTS = """
Persona {nombre edad}

ProcesarPersona(Clase p)
begin
    if (p ≠ NULL and p.edad > 18) then
    begin
        p.nombre ← "adulto"
    end
end
"""

# TEST 9: REPEAT-UNTIL
TEST_9_REPEAT = """
RepeatBasico()
begin
    i ← 0
    repeat
        i ← i + 1
    until (i ≥ 10)
end
"""

# TEST 10: Arrays multidimensionales
TEST_10_MATRIX = """
MultiplicarMatrices(A[][], B[][], n)
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

# TEST 11: Operadores booleanos (short-circuit)
TEST_11_BOOLEAN = """
BusquedaSegura(A[], n, x)
begin
    i ← 1
    encontrado ← F
    while (i ≤ n and not encontrado) do
    begin
        if (A[i] = x) then
        begin
            encontrado ← T
        end
        i ← i + 1
    end
    return encontrado
end
"""

# TEST 12: Funciones matemáticas
TEST_12_MATH = """
FuncionesMat(x, y)
begin
    a ← ceil(x / 2)
    b ← floor(y / 3)
    c ← x ^ 2 + y ^ 2
    return a + b + c
end
"""

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    """Ejecuta todos los tests"""
    
    print("="*70)
    print("PRUEBA DE GRAMÁTICA LARK - ANALIZADOR DE COMPLEJIDADES")
    print("="*70)
    
    # Cargar gramática
    parser = load_grammar()
    
    # Lista de tests
    tests = [
        (TEST_1_ASSIGNMENT, "Test 1: Asignación Simple"),
        (TEST_2_FOR, "Test 2: Ciclo FOR"),
        (TEST_3_WHILE, "Test 3: Ciclo WHILE"),
        (TEST_4_IF, "Test 4: IF-THEN-ELSE"),
        (TEST_5_ARRAY, "Test 5: Arrays"),
        (TEST_6_RECURSION, "Test 6: Recursión (Factorial)"),
        (TEST_7_BUBBLE_SORT, "Test 7: Bubble Sort"),
        (TEST_8_OBJECTS, "Test 8: Clases y Objetos"),
        (TEST_9_REPEAT, "Test 9: REPEAT-UNTIL"),
        (TEST_10_MATRIX, "Test 10: Matrices"),
        (TEST_11_BOOLEAN, "Test 11: Operadores Booleanos"),
        (TEST_12_MATH, "Test 12: Funciones Matemáticas"),
    ]
    
    # Ejecutar tests
    results = []
    for code, name in tests:
        success = test_parse(parser, code, name)
        results.append((name, success))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {name}")
    
    print("="*70)
    print(f"Resultado: {passed}/{total} tests pasados ({passed/total*100:.1f}%)")
    print("="*70)
    
    return passed == total

# ============================================================================
# MODO INTERACTIVO
# ============================================================================
def interactive_mode():
    """Modo interactivo para probar código personalizado"""
    
    print("\n" + "="*70)
    print("MODO INTERACTIVO")
    print("="*70)
    print("Ingresa tu código pseudocódigo (termina con una línea vacía):")
    print("Escribe 'salir' para terminar")
    print("="*70 + "\n")
    
    parser = load_grammar()
    
    while True:
        lines = []
        print("\n>>> Ingresa código (línea vacía para terminar):")
        
        while True:
            try:
                line = input()
                if line.strip() == '':
                    break
                if line.strip().lower() == 'salir':
                    print("¡Adiós!")
                    return
                lines.append(line)
            except EOFError:
                return
        
        if lines:
            code = '\n'.join(lines)
            test_parse(parser, code, "Código Personalizado")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(
        description="Prueba la gramática del analizador de complejidades"
    )
    arg_parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help="Modo interactivo para probar código personalizado"
    )
    arg_parser.add_argument(
        '--file', '-f',
        type=str,
        help="Parsear un archivo de pseudocódigo"
    )
    
    args = arg_parser.parse_args()
    
    if args.file:
        # Modo archivo
        parser = load_grammar()
        try:
            # Usar ruta absoluta o relativa al directorio actual
            import os
            if not os.path.isabs(args.file):
                file_path = os.path.abspath(args.file)
            else:
                file_path = args.file
                
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            test_parse(parser, code, f"Archivo: {args.file}")
        except FileNotFoundError:
            print(f"✗ Error: No se encontró el archivo '{args.file}'")
            print(f"   Ruta buscada: {file_path}")
            print(f"   Directorio actual: {os.getcwd()}")
            sys.exit(1)
    
    elif args.interactive:
        # Modo interactivo
        interactive_mode()
    
    else:
        # Modo tests automáticos
        success = main()
        sys.exit(0 if success else 1)