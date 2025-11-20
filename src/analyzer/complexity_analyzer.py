import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import sympy as sp

from syntax_tree.nodes import *


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class ComplexityResult:
    """Resultado del análisis de complejidad"""
    worst_case: str = "O(1)"
    best_case: str = "Ω(1)"
    average_case: str = "Θ(1)"
    exact_cost: Optional[str] = None
    explanation: str = ""
    steps: List[str] = field(default_factory=list)
    
    def __str__(self):
        return f"""
Complejidad:
  Peor caso:     {self.worst_case}
  Mejor caso:    {self.best_case}
  Caso promedio: {self.average_case}
  
Explicación: {self.explanation}
        """.strip()


# ============================================================================
# ANALIZADOR MEJORADO
# ============================================================================

class BasicComplexityAnalyzer:
    """
    Analizador de complejidad con soporte para ciclos dependientes.
    """
    
    def __init__(self):
        self.current_procedure = None
        self.results: Dict[str, ComplexityResult] = {}
        
        # Contexto para análisis de ciclos anidados
        self.loop_context: List[Dict] = []  # Stack de información de ciclos
    
    # ========================================================================
    # PUNTO DE ENTRADA
    # ========================================================================
    
    def analyze_program(self, program: ProgramNode) -> Dict[str, ComplexityResult]:
        """Analiza todo el programa"""
        for procedure in program.procedures:
            self.current_procedure = procedure.name
            result = self.analyze_procedure(procedure)
            self.results[procedure.name] = result
        
        return self.results
    
    def analyze_procedure(self, procedure: ProcedureNode) -> ComplexityResult:
        """Analiza un procedimiento completo"""
        result = self.analyze_block(procedure.body)
        
        params = ", ".join(p.name for p in procedure.parameters)
        result.explanation = f"Procedimiento {procedure.name}({params}): {result.explanation}"
        
        return result
    
    # ========================================================================
    # ANÁLISIS DE BLOQUES Y SENTENCIAS
    # ========================================================================
    
    def analyze_block(self, block: BlockNode) -> ComplexityResult:
        """Analiza un bloque de código"""
        if not block.statements:
            return ComplexityResult(explanation="Bloque vacío")
        
        results = [self.analyze_statement(stmt) for stmt in block.statements]
        return self._combine_sequential(results)
    
    def analyze_statement(self, stmt: StatementNode) -> ComplexityResult:
        """Despacha el análisis según el tipo de sentencia"""
        if isinstance(stmt, ForNode):
            return self.analyze_for(stmt)
        elif isinstance(stmt, WhileNode):
            return self.analyze_while(stmt)
        elif isinstance(stmt, RepeatNode):
            return self.analyze_repeat(stmt)
        elif isinstance(stmt, IfNode):
            return self.analyze_if(stmt)
        elif isinstance(stmt, AssignmentNode):
            return self.analyze_assignment(stmt)
        elif isinstance(stmt, CallStatementNode):
            return self.analyze_call(stmt)
        elif isinstance(stmt, ReturnNode):
            return ComplexityResult(explanation="Return statement")
        else:
            return ComplexityResult(explanation=f"Sentencia {type(stmt).__name__}")
    
    # ========================================================================
    # ANÁLISIS DE FOR MEJORADO
    # ========================================================================
    
    def analyze_for(self, node: ForNode) -> ComplexityResult:
        """
        Analiza un ciclo FOR con detección de dependencias.
        
        MEJORA CLAVE: Detecta si los límites dependen de variables
        de ciclos externos.
        """
        # 1. Guardar contexto del ciclo actual
        loop_info = {
            'variable': node.variable,
            'start': node.start,
            'end': node.end,
            'depth': len(self.loop_context)
        }
        self.loop_context.append(loop_info)
        
        # 2. Calcular número de iteraciones (considerando dependencias)
        iterations_symbolic = self._calculate_iterations_symbolic(node)
        iterations_simplified = self._simplify_iterations(iterations_symbolic)
        
        # 3. Detectar early exit
        has_early_exit = self._has_early_exit(node.body)
        
        # 4. Analizar el cuerpo
        body_result = self.analyze_block(node.body)
        
        # 5. Multiplicar: iteraciones × cuerpo
        result = self._multiply_complexity(iterations_simplified, body_result)
        
        # 6. Ajustar por early exit
        if has_early_exit:
            result.best_case = "Ω(1)"
            result.explanation = (
                f"FOR {node.variable} con early exit: "
                f"Peor caso {result.worst_case}, Mejor caso Ω(1)"
            )
        else:
            result.explanation = (
                f"FOR {node.variable}: {iterations_simplified} iteraciones × "
                f"{body_result.worst_case} = {result.worst_case}"
            )
        
        # 7. Agregar detalles
        result.steps.append(f"Ciclo FOR: {iterations_symbolic} → {iterations_simplified}")
        if has_early_exit:
            result.steps.append("Detectado early exit")
        
        # 8. Restaurar contexto
        self.loop_context.pop()
        
        return result
    
    # ========================================================================
    # CÁLCULO DE ITERACIONES SIMBÓLICO
    # ========================================================================
    
    def _calculate_iterations_symbolic(self, node: ForNode) -> str:
        """
        Calcula iteraciones usando análisis simbólico.
        
        CLAVE: Detecta patrones como:
        - for i=1 to n → n
        - for j=i to n → n-i+1
        - for j=i to n-i → n-2i+1 (ESTE ES EL CASO PROBLEMÁTICO)
        """
        # Extraer expresiones
        start_expr = self._expr_to_symbolic(node.start)
        end_expr = self._expr_to_symbolic(node.end)
        
        # Calcular: end - start + 1
        try:
            n = sp.Symbol('n', positive=True, integer=True)
            
            # Reemplazar identificadores por símbolos
            start_val = self._parse_symbolic(start_expr, n)
            end_val = self._parse_symbolic(end_expr, n)
            
            # Iteraciones = end - start + 1
            iterations = end_val - start_val + 1
            iterations = sp.simplify(iterations)
            
            return str(iterations)
        
        except Exception as e:
            # Fallback: usar heurística simple
            return self._calculate_iterations_heuristic(node.start, node.end)
    
    def _parse_symbolic(self, expr_str: str, n: sp.Symbol) -> sp.Expr:
        """
        Convierte una expresión string a SymPy.
        
        Ejemplos:
        - "n" → n
        - "n-1" → n-1
        - "i" → i (variable de ciclo externo)
        - "n-i" → n-i
        """
        # Detectar variables de ciclos externos
        for loop_info in self.loop_context:
            var_name = loop_info['variable']
            # Crear símbolo para la variable del ciclo
            if var_name in expr_str:
                var_symbol = sp.Symbol(var_name, positive=True, integer=True)
                expr_str = expr_str.replace(var_name, str(var_symbol))
        
        # Reemplazar 'n' con el símbolo n
        expr_str = expr_str.replace('n', str(n))
        
        # Parsear con SymPy
        try:
            return sp.sympify(expr_str)
        except:
            # Si falla, retornar la expresión como string
            return sp.Symbol(expr_str)
    
    def _expr_to_symbolic(self, expr: ExpressionNode) -> str:
        """Convierte un ExpressionNode a string simbólico"""
        if isinstance(expr, NumberNode):
            return str(expr.value)
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._expr_to_symbolic(expr.left)
            right = self._expr_to_symbolic(expr.right)
            
            # Manejar operadores
            if expr.op == '+':
                return f"({left}+{right})"
            elif expr.op == '-':
                return f"({left}-{right})"
            elif expr.op == '*':
                return f"({left}*{right})"
            elif expr.op == '/':
                return f"({left}/{right})"
            elif expr.op == '^':
                return f"({left}**{right})"
            else:
                return f"({left}{expr.op}{right})"
        else:
            return "expr"
    
    # ========================================================================
    # SIMPLIFICACIÓN DE ITERACIONES
    # ========================================================================
    
    def _simplify_iterations(self, symbolic_expr: str) -> str:
        """
        Simplifica una expresión simbólica de iteraciones.
        
        CASOS ESPECIALES:
        - "n - 2*i + 1" con i ∈ [1, n] → O(n) (promedio)
        - "n - i + 1" con i ∈ [1, n] → O(n)
        - Constantes → O(1)
        """
        try:
            n = sp.Symbol('n', positive=True, integer=True)
            expr = sp.sympify(symbolic_expr)
            
            # Caso 1: Constante
            if expr.is_number:
                return "1"
            
            # Caso 2: Lineal en n (sin variables de ciclo)
            if expr.free_symbols == {n}:
                # Simplificar: n-1 → n, n+1 → n, 2n → n
                coeff = expr.as_coefficient(n)
                if coeff is not None:
                    return "n"
                
                # Expresión más compleja pero lineal
                degree = sp.degree(expr, n)
                if degree == 1:
                    return "n"
                elif degree == 2:
                    return "n²"
                elif degree == 3:
                    return "n³"
            
            # Caso 3: Depende de variables de ciclos externos
            # Ejemplo: n - 2*i + 1
            # Necesitamos estimar el valor promedio
            if len(expr.free_symbols) > 1:
                # Detectar si tiene forma: n - k*i + c
                for loop_info in self.loop_context:
                    var_name = loop_info['variable']
                    var_symbol = sp.Symbol(var_name, positive=True, integer=True)
                    
                    if var_symbol in expr.free_symbols:
                        # Calcular suma total sobre el rango de i
                        # Ejemplo: Σ(n - 2i + 1) para i=1 to n
                        total = self._sum_over_range(expr, var_symbol, n)
                        
                        if total:
                            return self._classify_complexity(total, n)
            
            # Caso 4: No se pudo simplificar, retornar como está
            return str(expr)
        
        except Exception as e:
            # Fallback
            return symbolic_expr
    
    def _sum_over_range(self, expr: sp.Expr, var: sp.Symbol, n: sp.Symbol) -> Optional[sp.Expr]:
        """
        Calcula la suma de una expresión sobre un rango.
        
        Ejemplo:
        expr = n - 2*i + 1
        var = i
        rango = 1 to n-1
        
        Suma = Σ(n - 2i + 1) para i=1 to n-1
             = (n-1)*n - 2*(n-1)*n/2 + (n-1)
             = n² - n - n² + n + n - 1
             = n - 1 ≈ n
        """
        try:
            # Encontrar el rango de la variable (del contexto)
            start_val = 1  # Por defecto
            end_val = n - 1  # Por defecto para ciclos hasta n-1
            
            # Calcular la suma
            total = sp.summation(expr, (var, start_val, end_val))
            total = sp.simplify(total)
            
            return total
        
        except Exception as e:
            return None
    
    def _classify_complexity(self, expr: sp.Expr, n: sp.Symbol) -> str:
        """
        Clasifica la complejidad de una expresión.
        
        Ejemplos:
        - n → "n"
        - n² → "n²"
        - n - 1 → "n"
        - n²/2 → "n²"
        """
        # Simplificar primero
        expr = sp.simplify(expr)
        
        # Obtener el grado
        degree = sp.degree(expr, n)
        
        if degree == 0:
            return "1"
        elif degree == 1:
            return "n"
        elif degree == 2:
            return "n²"
        elif degree == 3:
            return "n³"
        else:
            return f"n^{degree}"
    
    # ========================================================================
    # HEURÍSTICA SIMPLE (FALLBACK)
    # ========================================================================
    
    def _calculate_iterations_heuristic(self, start: ExpressionNode, 
                                       end: ExpressionNode) -> str:
        """Heurística simple (usada como fallback)"""
        start_val = self._extract_value(start)
        end_val = self._extract_value(end)
        
        # Ambos constantes
        if isinstance(start_val, int) and isinstance(end_val, int):
            count = end_val - start_val + 1
            return str(max(1, count))
        
        # Start constante, end variable
        if isinstance(start_val, int):
            if start_val in [0, 1]:
                return self._simplify_expression(str(end_val))
            else:
                return "n"
        
        # Ambos variables
        return "n"
    
    def _extract_value(self, expr: ExpressionNode) -> Union[int, str]:
        """Extrae el valor de una expresión"""
        if isinstance(expr, NumberNode):
            return expr.value
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._extract_value(expr.left)
            right = self._extract_value(expr.right)
            
            if expr.op == "-":
                if isinstance(left, str) and isinstance(right, int):
                    return left
                elif isinstance(left, int) and isinstance(right, int):
                    return left - right
                else:
                    return f"{left}-{right}"
            elif expr.op == "+":
                if isinstance(left, str) and isinstance(right, int):
                    return left
                elif isinstance(left, int) and isinstance(right, int):
                    return left + right
                else:
                    return f"{left}+{right}"
            else:
                return f"({left}{expr.op}{right})"
        else:
            return "expr"
    
    # ========================================================================
    # MÉTODOS AUXILIARES (sin cambios significativos)
    # ========================================================================
    
    def _has_early_exit(self, block: BlockNode) -> bool:
        """Detecta si hay early exit (return en condicional)"""
        for stmt in block.statements:
            if isinstance(stmt, ReturnNode):
                return True
            elif isinstance(stmt, IfNode):
                if self._block_has_return(stmt.then_block):
                    return True
                if stmt.else_block and self._block_has_return(stmt.else_block):
                    return True
        return False
    
    def _block_has_return(self, block: BlockNode) -> bool:
        """Verifica si un bloque contiene un return"""
        for stmt in block.statements:
            if isinstance(stmt, ReturnNode):
                return True
            if isinstance(stmt, IfNode):
                if self._block_has_return(stmt.then_block):
                    return True
                if stmt.else_block and self._block_has_return(stmt.else_block):
                    return True
        return False
    
    def analyze_while(self, node: WhileNode) -> ComplexityResult:
        """Analiza WHILE"""
        body_result = self.analyze_block(node.body)
        has_early_exit = self._has_early_exit(node.body)
        
        if has_early_exit:
            return ComplexityResult(
                worst_case="O(n)",
                best_case="Ω(1)",
                average_case="Θ(n)",
                explanation="WHILE con early exit"
            )
        else:
            return ComplexityResult(
                worst_case="O(n)",
                best_case="Ω(1)",
                average_case="Θ(n)",
                explanation="WHILE con iteraciones indeterminadas"
            )
    
    def analyze_repeat(self, node: RepeatNode) -> ComplexityResult:
        """Analiza REPEAT-UNTIL"""
        body_result = self.analyze_block(node.body)
        return ComplexityResult(
            worst_case="O(n)",
            best_case="Ω(1)",
            average_case="Θ(n)",
            explanation="REPEAT-UNTIL"
        )
    
    def analyze_if(self, node: IfNode) -> ComplexityResult:
        """Analiza IF-THEN-ELSE"""
        then_result = self.analyze_block(node.then_block)
        
        if node.else_block:
            else_result = self.analyze_block(node.else_block)
            result = self._max_complexity(then_result, else_result)
            result.explanation = f"IF: max({then_result.worst_case}, {else_result.worst_case})"
        else:
            result = then_result
            result.explanation = f"IF: {then_result.worst_case}"
        
        return result
    
    def analyze_assignment(self, node: AssignmentNode) -> ComplexityResult:
        """Analiza asignación"""
        return ComplexityResult(explanation="Asignación: O(1)")
    
    def analyze_call(self, node: CallStatementNode) -> ComplexityResult:
        """Analiza llamada a procedimiento"""
        if node.name in self.results:
            called_result = self.results[node.name]
            return ComplexityResult(
                worst_case=called_result.worst_case,
                best_case=called_result.best_case,
                average_case=called_result.average_case,
                explanation=f"Llamada a {node.name}"
            )
        else:
            return ComplexityResult(explanation=f"Llamada a {node.name}: O(1)")
    
    # ========================================================================
    # COMBINACIÓN DE COMPLEJIDADES
    # ========================================================================
    
    def _combine_sequential(self, results: List[ComplexityResult]) -> ComplexityResult:
        """Combina complejidades de sentencias secuenciales"""
        if not results:
            return ComplexityResult()
        
        if len(results) == 1:
            result = results[0]
            result.worst_case = self._normalize_complexity(result.worst_case)
            result.best_case = self._normalize_complexity(result.best_case)
            result.average_case = self._normalize_complexity(result.average_case)
            return result
        
        # Encontrar la dominante
        dominant = results[0]
        for result in results[1:]:
            if self._is_greater_complexity(result.worst_case, dominant.worst_case):
                dominant = result
        
        dominant.worst_case = self._normalize_complexity(dominant.worst_case)
        dominant.best_case = self._normalize_complexity(dominant.best_case)
        dominant.average_case = self._normalize_complexity(dominant.average_case)
        
        return dominant
    
    def _multiply_complexity(self, iterations: str, 
                            body: ComplexityResult) -> ComplexityResult:
        """Multiplica complejidades"""
        body_order = self._extract_order(body.worst_case)
        iterations_simplified = self._simplify_expression(iterations)
        
        if body_order == "1":
            new_order = iterations_simplified
        elif iterations_simplified == "1":
            new_order = body_order
        else:
            new_order = self._multiply_orders(iterations_simplified, body_order)
        
        new_order = self._simplify_expression(new_order)
        
        return ComplexityResult(
            worst_case=self._normalize_complexity(f"O({new_order})"),
            best_case=self._normalize_complexity(f"Ω({new_order})"),
            average_case=self._normalize_complexity(f"Θ({new_order})"),
            explanation=f"{iterations} × {body.worst_case} = O({new_order})"
        )
    
    def _multiply_orders(self, a: str, b: str) -> str:
        """Multiplica dos órdenes"""
        if a == "1":
            return b
        if b == "1":
            return a
        
        a_power = self._get_n_power(a)
        b_power = self._get_n_power(b)
        
        if a_power is not None and b_power is not None:
            total_power = a_power + b_power
            
            if total_power == 1:
                return "n"
            elif total_power == 2:
                return "n²"
            elif total_power == 3:
                return "n³"
            else:
                return f"n^{total_power}"
        
        if "log" in a or "log" in b:
            if a_power == 1 or b_power == 1:
                return "n×log(n)"
        
        return f"{a}×{b}"
    
    def _get_n_power(self, expr: str) -> Optional[int]:
        """Obtiene la potencia de n"""
        expr = expr.strip()
        
        if expr == "n":
            return 1
        elif expr == "n²":
            return 2
        elif expr == "n³":
            return 3
        elif "n^" in expr:
            try:
                return int(expr.split("^")[1])
            except:
                return None
        else:
            return None
    
    def _simplify_expression(self, expr: str) -> str:
        """Simplifica una expresión"""
        expr = str(expr).strip()
        
        # Constantes → "1"
        try:
            int(expr)
            return "1"
        except:
            pass
        
        # Expresiones con n
        if 'n' in expr:
            if '-' in expr and not '×' in expr:
                return 'n'
            if '+' in expr and not '×' in expr:
                return 'n'
            if expr[0].isdigit() and 'n' in expr:
                return 'n'
            if '/' in expr and 'n' in expr:
                return 'n'
        
        # Ya es estándar
        if expr in ['1', 'n', 'n²', 'n³', 'log(n)', 'n×log(n)']:
            return expr
        
        # Multiplicaciones
        if '×' in expr:
            parts = expr.split('×')
            n_count = sum(1 for p in parts if 'n' in p)
            
            if n_count == 0:
                return '1'
            elif n_count == 1:
                return 'n'
            elif n_count == 2:
                return 'n²'
            elif n_count == 3:
                return 'n³'
        
        return expr
    
    def _normalize_complexity(self, complexity: str) -> str:
        """Normaliza una complejidad"""
        order = self._extract_order(complexity)
        simplified = self._simplify_expression(order)
        
        if complexity.startswith("O("):
            return f"O({simplified})"
        elif complexity.startswith("Ω("):
            return f"Ω({simplified})"
        elif complexity.startswith("Θ("):
            return f"Θ({simplified})"
        else:
            return complexity
    
    def _max_complexity(self, a: ComplexityResult, 
                        b: ComplexityResult) -> ComplexityResult:
        """Toma el máximo"""
        if self._is_greater_complexity(a.worst_case, b.worst_case):
            return a
        else:
            return b
    
    def _is_greater_complexity(self, a: str, b: str) -> bool:
        """Compara complejidades"""
        order = {
            "1": 0,
            "log(n)": 1,
            "n": 2,
            "n×log(n)": 3,
            "n²": 4,
            "n³": 5,
            "2^n": 6,
            "n!": 7
        }
        
        a_order = self._extract_order(a)
        b_order = self._extract_order(b)
        
        return order.get(a_order, 2) > order.get(b_order, 2)
    
    def _extract_order(self, complexity: str) -> str:
        """Extrae el orden"""
        for prefix in ["O(", "Ω(", "Θ("]:
            if complexity.startswith(prefix):
                complexity = complexity[len(prefix):-1]
        
        return complexity
    
    def _expr_to_str(self, expr: ExpressionNode) -> str:
        """Convierte expresión a string"""
        if isinstance(expr, NumberNode):
            return str(expr.value)
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._expr_to_str(expr.left)
            right = self._expr_to_str(expr.right)
            return f"{left}{expr.op}{right}"
        else:
            return "expr"


# ============================================================================
# API PÚBLICA
# ============================================================================

def analyze_complexity(ast: ProgramNode) -> Dict[str, ComplexityResult]:
    """Analiza complejidad de un programa completo"""
    analyzer = BasicComplexityAnalyzer()
    return analyzer.analyze_program(ast)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    from parser.parser import parse
    
    # CASO PROBLEMÁTICO
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
    
    print("="*70)
    print("ANÁLISIS MEJORADO - CASO PROBLEMÁTICO")
    print("="*70)
    print("\nCódigo:")
    print(code)
    
    # Parse
    ast = parse(code)
    
    # Analyze
    results = analyze_complexity(ast)
    
    # Display
    for proc_name, result in results.items():
        print("\n" + "="*70)
        print(f"Procedimiento: {proc_name}")
        print("="*70)
        print(result)
        
        if result.steps:
            print("\nPasos del análisis:")
            for step in result.steps:
                print(f"  • {step}")