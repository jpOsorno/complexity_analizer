"""
Herramienta de Diagnóstico AST
==============================

Identifica Tree y Token sin transformar en el AST
Ejecutar: python diagnostic_tool.py
"""

import sys
import os
from lark import Token, Tree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.parser.parser import parse


def diagnose_ast(ast, path="root", depth=0):
    """
    Recorre el AST y reporta cualquier Tree o Token sin transformar.
    
    Args:
        ast: Nodo del AST a examinar
        path: Ruta actual en el árbol
        depth: Profundidad actual (para indentación)
    
    Returns:
        List of (path, type, value) de problemas encontrados
    """
    issues = []
    indent = "  " * depth
    
    # Verificar tipo del nodo actual
    node_type = type(ast).__name__
    
    # PROBLEMA 1: Es un Tree
    if isinstance(ast, Tree):
        print(f"{indent}❌ {path}: Tree({ast.data})")
        issues.append((path, "Tree", ast.data))
        return issues
    
    # PROBLEMA 2: Es un Token
    if isinstance(ast, Token):
        print(f"{indent}❌ {path}: Token({ast.type}={ast.value})")
        issues.append((path, "Token", f"{ast.type}={ast.value}"))
        return issues
    
    # OK: Es un nodo del AST
    if hasattr(ast, '__dict__'):
        print(f"{indent}✓ {path}: {node_type}")
        
        # Examinar todos los atributos
        for attr_name, attr_value in ast.__dict__.items():
            attr_path = f"{path}.{attr_name}"
            
            if attr_value is None:
                continue
            
            # Lista de nodos
            elif isinstance(attr_value, list):
                for i, item in enumerate(attr_value):
                    item_path = f"{attr_path}[{i}]"
                    issues.extend(diagnose_ast(item, item_path, depth + 1))
            
            # Nodo individual
            elif hasattr(attr_value, '__dict__'):
                issues.extend(diagnose_ast(attr_value, attr_path, depth + 1))
            
            # Valor primitivo (OK)
            else:
                print(f"{indent}  • {attr_name}: {type(attr_value).__name__} = {attr_value}")
    
    # No es un nodo del AST (valor primitivo)
    else:
        print(f"{indent}• {path}: {node_type} = {ast}")
    
    return issues


def main():
    """Función principal"""
    
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
    
    print("="*70)
    print("DIAGNÓSTICO DEL AST")
    print("="*70)
    print("\nCódigo:")
    print(code)
    print("\n" + "="*70)
    print("ESTRUCTURA DEL AST")
    print("="*70)
    print()
    
    try:
        # Parsear
        ast = parse(code)
        
        # Diagnosticar
        issues = diagnose_ast(ast)
        
        # Resumen
        print("\n" + "="*70)
        print("RESUMEN")
        print("="*70)
        
        if issues:
            print(f"\n❌ Encontrados {len(issues)} problemas:\n")
            for path, issue_type, value in issues:
                print(f"  • {path}")
                print(f"    Tipo: {issue_type}")
                print(f"    Valor: {value}")
                print()
            
            print("ACCIÓN REQUERIDA:")
            print("  1. Reemplazar src/ast/transformer.py con la versión corregida")
            print("  2. Verificar que visit_tokens=True en __init__")
            print("  3. Asegurar que todos los métodos usan _ensure_node()")
        else:
            print("\n✅ ¡No se encontraron problemas!")
            print("   El AST está limpio (sin Tree ni Token)")
            print()
            print("   Puedes ejecutar:")
            print("   • python -m src.analyzer.visitor")
            print("   • python demo.py")
    
    except Exception as e:
        print(f"\n❌ ERROR durante el parsing:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()