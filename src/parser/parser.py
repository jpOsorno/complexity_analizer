"""
Parser Wrapper: API limpia para parsear pseudocódigo
====================================================

Este módulo encapsula Lark + Transformer y expone una interfaz simple:
    parse(source_code: str) -> ProgramNode

Maneja:
- Carga de gramática (singleton pattern)
- Parsing con Lark
- Transformación a AST personalizado
- Manejo de errores con mensajes claros
"""

import os
from typing import Optional
from lark import Lark, LarkError
from lark.exceptions import UnexpectedInput, UnexpectedToken

# Importar nuestro AST y transformer
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ..syntax_tree.nodes import ProgramNode
from ..syntax_tree.transformer import transform_to_ast


# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class ParseError(Exception):
    """Error durante el parsing de pseudocódigo"""
    
    def __init__(self, message: str, line: Optional[int] = None, 
                 column: Optional[int] = None, context: Optional[str] = None):
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Formatea el mensaje de error con contexto"""
        msg = f"Error de sintaxis: {self.message}"
        
        if self.line is not None:
            msg += f"\n  Línea {self.line}"
            if self.column is not None:
                msg += f", Columna {self.column}"
        
        if self.context:
            msg += f"\n  Contexto: {self.context}"
        
        return msg


# ============================================================================
# PARSER SINGLETON
# ============================================================================

class PseudocodeParser:
    """
    Parser singleton para pseudocódigo.
    
    Carga la gramática una sola vez y la reutiliza para todos los parseos.
    Thread-safe para múltiples requests simultáneos.
    """
    
    _instance: Optional['PseudocodeParser'] = None
    _parser: Optional[Lark] = None
    
    def __new__(cls):
        """Implementa el patrón Singleton"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Inicializa el parser de Lark (solo una vez)"""
        try:
            # Obtener ruta de la gramática
            parser_dir = os.path.dirname(os.path.abspath(__file__))
            grammar_path = os.path.join(parser_dir, 'grammar.lark')
            
            # Cargar gramática
            with open(grammar_path, 'r', encoding='utf-8') as f:
                grammar = f.read()
            
            # Crear parser Lark con configuración óptima
            self._parser = Lark(
                grammar,
                parser='earley',              # Parser robusto
                start='program',              # Símbolo inicial
                propagate_positions=True,     # Para errores detallados
                maybe_placeholders=False,     # Errores más claros
                ambiguity='explicit'          # Detectar ambigüedades
            )
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No se encontró grammar.lark en {grammar_path}"
            )
        except Exception as e:
            raise RuntimeError(f"Error al cargar gramática: {e}")
    
    def parse(self, source_code: str) -> ProgramNode:
        """
        Parsea código pseudocódigo y retorna el AST.
        
        Args:
            source_code: Código fuente en pseudocódigo
            
        Returns:
            ProgramNode: Raíz del AST
            
        Raises:
            ParseError: Si hay errores de sintaxis
            
        Example:
            >>> parser = PseudocodeParser()
            >>> ast = parser.parse('''
            ... Factorial(n)
            ... begin
            ...     if (n ≤ 1) then
            ...         return 1
            ...     else
            ...         return n * call Factorial(n-1)
            ... end
            ... ''')
            >>> print(ast.procedures[0].name)
            Factorial
        """
        if not source_code or not source_code.strip():
            raise ParseError("El código fuente está vacío")
        
        try:
            # 1. Parsear con Lark
            lark_tree = self._parser.parse(source_code)
            
            # 2. Transformar a nuestro AST
            ast = transform_to_ast(lark_tree)
            
            # 3. Validación básica
            if not isinstance(ast, ProgramNode):
                raise ParseError("El resultado no es un programa válido")
            
            if not ast.procedures:
                raise ParseError("El programa no contiene procedimientos")
            
            return ast
        
        except UnexpectedToken as e:
            # Error: token inesperado
            raise ParseError(
                message=f"Token inesperado: '{e.token}'",
                line=e.line,
                column=e.column,
                context=self._get_context(source_code, e.line)
            )
        
        except UnexpectedInput as e:
            # Error: entrada inesperada
            raise ParseError(
                message="Entrada inesperada en el código",
                line=getattr(e, 'line', None),
                column=getattr(e, 'column', None),
                context=self._get_context(source_code, getattr(e, 'line', None))
            )
        
        except LarkError as e:
            # Otros errores de Lark
            raise ParseError(
                message=str(e),
                context=source_code[:100] if len(source_code) > 100 else source_code
            )
        
        except Exception as e:
            # Error inesperado
            raise ParseError(
                message=f"Error inesperado durante el parsing: {e}",
                context=source_code[:100] if len(source_code) > 100 else source_code
            )
    
    def _get_context(self, source: str, line: Optional[int], 
                     context_lines: int = 2) -> Optional[str]:
        """
        Obtiene líneas de contexto alrededor de un error.
        
        Args:
            source: Código fuente completo
            line: Número de línea del error (1-indexed)
            context_lines: Cuántas líneas mostrar antes/después
            
        Returns:
            String con el contexto, o None si no hay línea
        """
        if line is None:
            return None
        
        lines = source.split('\n')
        start = max(0, line - context_lines - 1)
        end = min(len(lines), line + context_lines)
        
        context = []
        for i in range(start, end):
            marker = ">>> " if i == line - 1 else "    "
            context.append(f"{marker}{i+1:4d} | {lines[i]}")
        
        return '\n'.join(context)


# ============================================================================
# API PÚBLICA: Función de conveniencia
# ============================================================================

def parse(source_code: str) -> ProgramNode:
    """
    Parsea pseudocódigo y retorna el AST.
    
    Esta es la función principal que debe usar el resto del sistema.
    
    Args:
        source_code: Código fuente en pseudocódigo
        
    Returns:
        ProgramNode: Raíz del AST
        
    Raises:
        ParseError: Si hay errores de sintaxis
        
    Example:
        >>> from src.parser.parser import parse
        >>> ast = parse('''
        ... BubbleSort(A[], n)
        ... begin
        ...     for i ← 1 to n-1 do
        ...     begin
        ...         for j ← 1 to n-i do
        ...         begin
        ...             if (A[j] > A[j+1]) then
        ...             begin
        ...                 temp ← A[j]
        ...                 A[j] ← A[j+1]
        ...                 A[j+1] ← temp
        ...             end
        ...         end
        ...     end
        ... end
        ... ''')
        >>> print(f"Procedimiento: {ast.procedures[0].name}")
        Procedimiento: BubbleSort
    """
    parser = PseudocodeParser()
    return parser.parse(source_code)


# ============================================================================
# UTILIDADES: Validación y Pretty-Printing
# ============================================================================

def validate_syntax(source_code: str) -> tuple[bool, Optional[str]]:
    """
    Valida la sintaxis sin retornar el AST.
    
    Args:
        source_code: Código a validar
        
    Returns:
        (is_valid, error_message): Tupla con resultado y mensaje de error
        
    Example:
        >>> valid, error = validate_syntax("Simple() begin x ← 5 end")
        >>> print(valid)
        True
    """
    try:
        parse(source_code)
        return (True, None)
    except ParseError as e:
        return (False, str(e))
    except Exception as e:
        return (False, f"Error inesperado: {e}")


def parse_file(filepath: str) -> ProgramNode:
    """
    Parsea un archivo de pseudocódigo.
    
    Args:
        filepath: Ruta al archivo
        
    Returns:
        ProgramNode: AST del programa
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ParseError: Si hay errores de sintaxis
        
    Example:
        >>> ast = parse_file("examples/fibonacci.txt")
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        return parse(source_code)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")


# ============================================================================
# CLI: Modo de prueba desde línea de comandos
# ============================================================================

def main():
    """CLI para probar el parser desde la terminal"""
    import argparse
    
    parser_cli = argparse.ArgumentParser(
        description="Parser de pseudocódigo para análisis de complejidad"
    )
    parser_cli.add_argument(
        'file',
        nargs='?',
        help="Archivo de pseudocódigo a parsear"
    )
    parser_cli.add_argument(
        '--validate', '-v',
        action='store_true',
        help="Solo validar sintaxis (no mostrar AST)"
    )
    parser_cli.add_argument(
        '--interactive', '-i',
        action='store_true',
        help="Modo interactivo"
    )
    
    args = parser_cli.parse_args()
    
    if args.interactive:
        print("=== MODO INTERACTIVO ===")
        print("Ingresa pseudocódigo (Ctrl+D para terminar):\n")
        
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        source = '\n'.join(lines)
    
    elif args.file:
        try:
            ast = parse_file(args.file)
            source = open(args.file).read()
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    else:
        # Código de ejemplo por defecto
        source = """
Simple()
begin
    x ← 5
    y ← x + 3
end
        """
    
    # Validar o parsear
    if args.validate:
        valid, error = validate_syntax(source)
        if valid:
            print("✓ Sintaxis válida")
            return 0
        else:
            print(f"✗ Error de sintaxis:\n{error}")
            return 1
    else:
        try:
            ast = parse(source)
            print("✓ Parsing exitoso\n")
            print(f"Programa con {len(ast.procedures)} procedimiento(s):")
            for proc in ast.procedures:
                params = ', '.join(p.name for p in proc.parameters)
                print(f"  - {proc.name}({params})")
            return 0
        except ParseError as e:
            print(f"✗ {e}")
            return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())