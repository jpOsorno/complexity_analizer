"""
Analizador de Complejidad Básico (Sin Recursión)
================================================

Este módulo implementa el análisis de complejidad temporal para
algoritmos iterativos (FOR, WHILE, REPEAT).

Calcula:
- O (peor caso)
- Ω (mejor caso)  
- Θ (caso promedio)

NO maneja recursión (eso será en complexity_analyzer_recursive.py)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# Importar desde syntax_tree en lugar de ast
from syntax_tree.nodes import *


# ============================================================================
# RESULTADO DEL ANÁLISIS
# ============================================================================

@dataclass
class ComplexityResult:
    """
    Resultado del análisis de complejidad de un nodo/algoritmo.
    
    Attributes:
        worst_case: Complejidad en el peor caso (notación O)
        best_case: Complejidad en el mejor caso (notación Ω)
        average_case: Complejidad en el caso promedio (notación Θ)
        exact_cost: Expresión matemática exacta (usando sympy)
        explanation: Explicación en lenguaje natural
        steps: Pasos detallados del análisis
    """
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
# ANALIZADOR BÁSICO (ITERATIVO)
# ============================================================================

class BasicComplexityAnalyzer:
    """
    Analizador de complejidad para algoritmos iterativos.
    
    Analiza:
    - Ciclos FOR
    - Ciclos WHILE (con límites detectables)
    - Ciclos REPEAT
    - Condicionales IF
    - Sentencias simples (asignaciones, llamadas)
    
    NO analiza:
    - Recursión (usar ComplexityAnalyzerRecursive)
    - Ciclos con límites no determinables
    """
    
    def __init__(self):
        self.current_procedure = None
        self.results: Dict[str, ComplexityResult] = {}
        
    # ========================================================================
    # PUNTO DE ENTRADA
    # ========================================================================
    
    def analyze_program(self, program: ProgramNode) -> Dict[str, ComplexityResult]:
        """
        Analiza todo el programa y retorna complejidades por procedimiento.
        
        Args:
            program: Nodo raíz del AST
            
        Returns:
            Diccionario {nombre_procedimiento: ComplexityResult}
        """
        for procedure in program.procedures:
            self.current_procedure = procedure.name
            result = self.analyze_procedure(procedure)
            self.results[procedure.name] = result
        
        return self.results
    
    def analyze_procedure(self, procedure: ProcedureNode) -> ComplexityResult:
        """
        Analiza un procedimiento completo.
        
        Args:
            procedure: Nodo del procedimiento
            
        Returns:
            ComplexityResult con las complejidades
        """
        result = self.analyze_block(procedure.body)
        
        # Agregar información del procedimiento
        params = ", ".join(p.name for p in procedure.parameters)
        result.explanation = f"Procedimiento {procedure.name}({params}): {result.explanation}"
        
        return result
    
    # ========================================================================
    # ANÁLISIS DE BLOQUES Y SENTENCIAS
    # ========================================================================
    
    def analyze_block(self, block: BlockNode) -> ComplexityResult:
        """
        Analiza un bloque de código.
        
        La complejidad de un bloque es la SUMA de las complejidades
        de sus sentencias (se ejecutan secuencialmente).
        """
        if not block.statements:
            return ComplexityResult(
                worst_case="O(1)",
                best_case="Ω(1)",
                average_case="Θ(1)",
                explanation="Bloque vacío"
            )
        
        # Analizar cada sentencia
        results = [self.analyze_statement(stmt) for stmt in block.statements]
        
        # Combinar resultados (tomar el dominante)
        return self._combine_sequential(results)
    
    def analyze_statement(self, stmt: StatementNode) -> ComplexityResult:
        """
        Despacha el análisis según el tipo de sentencia.
        """
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
            return ComplexityResult(
                worst_case="O(1)",
                best_case="Ω(1)",
                average_case="Θ(1)",
                explanation="Return statement"
            )
        else:
            # Sentencia desconocida, asumir O(1)
            return ComplexityResult(explanation=f"Sentencia {type(stmt).__name__}")
    
    # ========================================================================
    # ANÁLISIS DE CICLOS FOR
    # ========================================================================
    
    def analyze_for(self, node: ForNode) -> ComplexityResult:
        """
        Analiza un ciclo FOR.
        
        for variable ← start to end do
            body
        
        Complejidad: O(iteraciones × cuerpo)
        
        IMPORTANTE: Detecta early exit (return dentro del ciclo)
        """
        # 1. Calcular número de iteraciones
        iterations = self._calculate_for_iterations(node.start, node.end)
        
        # 2. Detectar si hay early exit (return en el cuerpo)
        has_early_exit = self._has_early_exit(node.body)
        
        # 3. Analizar el cuerpo
        body_result = self.analyze_block(node.body)
        
        # 4. Multiplicar: O(iteraciones × cuerpo)
        result = self._multiply_complexity(iterations, body_result)
        
        # 5. Ajustar mejor caso si hay early exit
        if has_early_exit:
            # Mejor caso: puede terminar en la primera iteración
            result.best_case = "Ω(1)"
            result.average_case = f"Θ({self._simplify_expression(iterations)})"
            
            result.explanation = (
                f"FOR {node.variable} = {self._expr_to_str(node.start)} "
                f"to {self._expr_to_str(node.end)} con early exit: "
                f"Peor caso {result.worst_case}, Mejor caso {result.best_case}"
            )
        else:
            # Sin early exit: siempre se ejecuta completo
            result.explanation = (
                f"FOR {node.variable} = {self._expr_to_str(node.start)} "
                f"to {self._expr_to_str(node.end)}: "
                f"{iterations} iteraciones × {body_result.worst_case} = {result.worst_case}"
            )
        
        result.steps.append(f"Ciclo FOR con {iterations} iteraciones")
        if has_early_exit:
            result.steps.append(f"Detectado early exit (return en el cuerpo)")
            result.steps.append(f"Mejor caso: Ω(1) (primera iteración)")
            result.steps.append(f"Peor caso: {result.worst_case} (última iteración)")
        else:
            result.steps.append(f"Cuerpo: {body_result.worst_case}")
            result.steps.append(f"Total: {result.worst_case}")
        
        return result
    
    def _has_early_exit(self, block: BlockNode) -> bool:
        """
        Detecta si un bloque tiene un early exit (return dentro de un condicional).
        
        Casos que detecta:
        - if (...) then return
        - if (...) then begin return ... end
        """
        for stmt in block.statements:
            if isinstance(stmt, ReturnNode):
                # Return directo (raro en ciclos, pero posible)
                return True
            elif isinstance(stmt, IfNode):
                # Return dentro de un IF
                if self._block_has_return(stmt.then_block):
                    return True
                if stmt.else_block and self._block_has_return(stmt.else_block):
                    return True
        
        return False
    
    def _block_has_return(self, block: BlockNode) -> bool:
        """
        Verifica si un bloque contiene un return.
        """
        for stmt in block.statements:
            if isinstance(stmt, ReturnNode):
                return True
            # Buscar recursivamente en bloques anidados
            if isinstance(stmt, IfNode):
                if self._block_has_return(stmt.then_block):
                    return True
                if stmt.else_block and self._block_has_return(stmt.else_block):
                    return True
        
        return False
    
    def _calculate_for_iterations(self, start: ExpressionNode, 
                                   end: ExpressionNode) -> str:
        """
        Calcula el número de iteraciones de un FOR.
        
        Casos comunes:
        - for i ← 1 to n → n iteraciones
        - for i ← 1 to n-1 → n-1 ≈ n iteraciones
        - for i ← 0 to n → n+1 ≈ n iteraciones
        - for i ← 1 to 10 → 10 iteraciones (constante)
        """
        # Extraer expresiones
        start_val = self._extract_value(start)
        end_val = self._extract_value(end)
        
        # Caso 1: Ambos son constantes
        if isinstance(start_val, int) and isinstance(end_val, int):
            count = end_val - start_val + 1
            return str(max(1, count))
        
        # Caso 2: Start es constante, end es variable
        if isinstance(start_val, int):
            if start_val == 0:
                # 0 to n → n+1 ≈ n
                return "n"
            elif start_val == 1:
                # 1 to n → n
                # 1 to n-1 → n-1 ≈ n
                return self._simplify_expression(str(end_val))
            else:
                # k to n → n-k+1 ≈ n (si k es pequeño)
                return "n"
        
        # Caso 3: Ambos son variables/expresiones
        # end - start + 1 ≈ asumimos que es proporcional a n
        return "n"
    
    def _extract_value(self, expr: ExpressionNode) -> Union[int, str]:
        """
        Extrae el valor de una expresión.
        
        Returns:
            int si es constante, str si es variable/expresión
        """
        if isinstance(expr, NumberNode):
            return expr.value
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._extract_value(expr.left)
            right = self._extract_value(expr.right)
            
            # Simplificar expresiones comunes
            if expr.op == "-":
                if isinstance(left, str) and isinstance(right, int):
                    # n-1, n-2, etc. → simplemente 'n' para Big-O
                    return left
                elif isinstance(left, int) and isinstance(right, int):
                    return left - right
                else:
                    return f"{left}-{right}"
                    
            elif expr.op == "+":
                if isinstance(left, str) and isinstance(right, int):
                    # n+1, n+2, etc. → simplemente 'n' para Big-O
                    return left
                elif isinstance(left, int) and isinstance(right, int):
                    return left + right
                else:
                    return f"{left}+{right}"
                    
            elif expr.op == "*":
                if isinstance(left, int) and isinstance(right, int):
                    return left * right
                elif isinstance(left, int) and isinstance(right, str):
                    # 2*n → 'n' para Big-O
                    return right
                elif isinstance(left, str) and isinstance(right, int):
                    # n*2 → 'n' para Big-O
                    return left
                else:
                    return f"{left}*{right}"
                    
            elif expr.op == "/":
                if isinstance(left, int) and isinstance(right, int):
                    return left // right
                elif isinstance(left, str):
                    # n/2 → 'n' para Big-O
                    return left
                else:
                    return f"{left}/{right}"
            
            # Otros operadores
            return f"({left}{expr.op}{right})"
        else:
            return "expr"
    
    # ========================================================================
    # ANÁLISIS DE CICLOS WHILE Y REPEAT
    # ========================================================================
    
    def analyze_while(self, node: WhileNode) -> ComplexityResult:
        """
        Analiza un ciclo WHILE.
        
        while (condition) do body
        
        PROBLEMA: No siempre podemos determinar cuántas iteraciones.
        Heurística: buscar patrones comunes.
        """
        # Analizar el cuerpo
        body_result = self.analyze_block(node.body)
        
        # Detectar early exit
        has_early_exit = self._has_early_exit(node.body)
        
        # Intentar detectar el patrón de iteración
        iterations = self._detect_while_iterations(node)
        
        if iterations:
            result = self._multiply_complexity(iterations, body_result)
            
            if has_early_exit:
                result.best_case = "Ω(1)"
                result.explanation = (
                    f"WHILE con ~{iterations} iteraciones × {body_result.worst_case} "
                    f"(early exit detectado)"
                )
            else:
                result.explanation = (
                    f"WHILE con ~{iterations} iteraciones × {body_result.worst_case}"
                )
        else:
            # No podemos determinar, asumir O(n)
            if has_early_exit:
                result = ComplexityResult(
                    worst_case="O(n)",
                    best_case="Ω(1)",  # Early exit en primera iteración
                    average_case="Θ(n)",
                    explanation=f"WHILE con early exit: Mejor caso Ω(1), Peor caso O(n)"
                )
            else:
                result = ComplexityResult(
                    worst_case="O(n)",
                    best_case="Ω(1)",  # Podría no ejecutarse
                    average_case="Θ(n)",
                    explanation=f"WHILE con número indeterminado de iteraciones"
                )
        
        return result
    
    def analyze_repeat(self, node: RepeatNode) -> ComplexityResult:
        """
        Analiza un ciclo REPEAT-UNTIL.
        
        repeat body until (condition)
        
        Similar a WHILE pero se ejecuta al menos una vez.
        """
        body_result = self.analyze_block(node.body)
        
        # Repeat se ejecuta AL MENOS una vez
        result = ComplexityResult(
            worst_case="O(n)",
            best_case="Ω(1)",  # Al menos una iteración
            average_case="Θ(n)",
            explanation=f"REPEAT-UNTIL con cuerpo {body_result.worst_case}"
        )
        
        return result
    
    def _detect_while_iterations(self, node: WhileNode) -> Optional[str]:
        """
        Intenta detectar cuántas veces se ejecuta un WHILE.
        
        Patrones comunes:
        - while (i < n) con i++ → n iteraciones
        - while (left <= right) con búsqueda binaria → log(n)
        """
        # Por ahora, retornar None (no determinable)
        # TODO: Implementar detección de patrones
        return None
    
    # ========================================================================
    # ANÁLISIS DE CONDICIONALES
    # ========================================================================
    
    def analyze_if(self, node: IfNode) -> ComplexityResult:
        """
        Analiza un condicional IF-THEN-ELSE.
        
        La complejidad es el MÁXIMO entre las dos ramas.
        """
        # Analizar rama THEN
        then_result = self.analyze_block(node.then_block)
        
        # Analizar rama ELSE (si existe)
        if node.else_block:
            else_result = self.analyze_block(node.else_block)
            
            # Tomar el máximo
            result = self._max_complexity(then_result, else_result)
            result.explanation = (
                f"IF: max(THEN: {then_result.worst_case}, "
                f"ELSE: {else_result.worst_case}) = {result.worst_case}"
            )
        else:
            # Solo THEN
            result = then_result
            result.explanation = f"IF: {then_result.worst_case}"
        
        # Agregar costo de la condición (O(1))
        result.steps.append("Evaluación de condición: O(1)")
        result.steps.extend(then_result.steps)
        
        return result
    
    # ========================================================================
    # ANÁLISIS DE SENTENCIAS SIMPLES
    # ========================================================================
    
    def analyze_assignment(self, node: AssignmentNode) -> ComplexityResult:
        """
        Analiza una asignación.
        
        x ← expr
        
        Complejidad: O(1) si expr es simple
        """
        # Analizar el lado derecho
        expr_cost = self._analyze_expression(node.value)
        
        # Simplificar el costo
        expr_cost_simplified = self._simplify_expression(expr_cost)
        
        return ComplexityResult(
            worst_case=self._normalize_complexity(f"O({expr_cost_simplified})"),
            best_case=self._normalize_complexity(f"Ω({expr_cost_simplified})"),
            average_case=self._normalize_complexity(f"Θ({expr_cost_simplified})"),
            explanation=f"Asignación: {self._normalize_complexity(f'O({expr_cost_simplified})')}"
        )
    
    def analyze_call(self, node: CallStatementNode) -> ComplexityResult:
        """
        Analiza una llamada a procedimiento.
        
        call Procedimiento(args)
        
        PROBLEMA: Necesitamos conocer la complejidad del procedimiento llamado.
        Por ahora, asumir O(1) o buscar en resultados previos.
        """
        # Buscar si ya analizamos este procedimiento
        if node.name in self.results:
            called_result = self.results[node.name]
            return ComplexityResult(
                worst_case=called_result.worst_case,
                best_case=called_result.best_case,
                average_case=called_result.average_case,
                explanation=f"Llamada a {node.name}: {called_result.worst_case}"
            )
        else:
            # No conocemos la complejidad, asumir O(1)
            return ComplexityResult(
                explanation=f"Llamada a {node.name} (complejidad desconocida, asumido O(1))"
            )
    
    def _analyze_expression(self, expr: ExpressionNode) -> str:
        """
        Analiza la complejidad de evaluar una expresión.
        
        La mayoría de expresiones son O(1).
        Excepciones:
        - Llamadas a funciones
        - Accesos complejos a estructuras
        """
        if isinstance(expr, FunctionCallNode):
            # Llamada a función, podría ser costosa
            return "f(n)"  # Placeholder
        else:
            # Expresión aritmética/booleana simple
            return "1"
    
    # ========================================================================
    # COMBINACIÓN DE COMPLEJIDADES
    # ========================================================================
    
    def _combine_sequential(self, results: List[ComplexityResult]) -> ComplexityResult:
        """
        Combina complejidades de sentencias secuenciales.
        
        Tomar la complejidad DOMINANTE (la mayor).
        """
        if not results:
            return ComplexityResult()
        
        if len(results) == 1:
            # Normalizar antes de retornar
            result = results[0]
            result.worst_case = self._normalize_complexity(result.worst_case)
            result.best_case = self._normalize_complexity(result.best_case)
            result.average_case = self._normalize_complexity(result.average_case)
            return result
        
        # Encontrar la complejidad dominante
        dominant = results[0]
        for result in results[1:]:
            if self._is_greater_complexity(result.worst_case, dominant.worst_case):
                dominant = result
        
        # Normalizar
        dominant.worst_case = self._normalize_complexity(dominant.worst_case)
        dominant.best_case = self._normalize_complexity(dominant.best_case)
        dominant.average_case = self._normalize_complexity(dominant.average_case)
        
        # Combinar explicaciones
        explanations = [r.explanation for r in results if r.explanation]
        dominant.explanation = " + ".join(explanations) + f" → {dominant.worst_case}"
        
        return dominant
    
    def _multiply_complexity(self, iterations: str, 
                            body: ComplexityResult) -> ComplexityResult:
        """
        Multiplica complejidades: iteraciones × cuerpo.
        
        Ejemplo: n iteraciones × O(n) cuerpo = O(n²)
        """
        # Extraer el orden del cuerpo
        body_order = self._extract_order(body.worst_case)
        
        # Simplificar las iteraciones primero
        iterations_simplified = self._simplify_expression(iterations)
        
        # Multiplicar
        if body_order == "1":
            new_order = iterations_simplified
        elif iterations_simplified == "1":
            new_order = body_order
        else:
            # Multiplicar simbólicamente
            new_order = self._multiply_orders(iterations_simplified, body_order)
        
        # Simplificar el resultado final
        new_order = self._simplify_expression(new_order)
        
        # Normalizar las notaciones
        worst_normalized = self._normalize_complexity(f"O({new_order})")
        best_normalized = self._normalize_complexity(f"Ω({new_order})")
        avg_normalized = self._normalize_complexity(f"Θ({new_order})")
        
        return ComplexityResult(
            worst_case=worst_normalized,
            best_case=best_normalized,
            average_case=avg_normalized,
            explanation=f"{iterations} × {body.worst_case} = {worst_normalized}"
        )
    
    def _multiply_orders(self, a: str, b: str) -> str:
        """
        Multiplica dos órdenes de complejidad.
        
        Ejemplos:
        - n × n = n²
        - n × n² = n³
        - n × 1 = n
        - log(n) × n = n×log(n)
        """
        # Casos base
        if a == "1":
            return b
        if b == "1":
            return a
        
        # Detectar potencias de n
        a_power = self._get_n_power(a)
        b_power = self._get_n_power(b)
        
        if a_power is not None and b_power is not None:
            # Ambos son potencias de n
            total_power = a_power + b_power
            
            if total_power == 1:
                return "n"
            elif total_power == 2:
                return "n²"
            elif total_power == 3:
                return "n³"
            elif total_power == 4:
                return "n⁴"
            else:
                return f"n^{total_power}"
        
        # Si uno tiene log(n)
        if "log" in a or "log" in b:
            # n × log(n)
            if a_power == 1 or b_power == 1:
                return "n×log(n)"
            elif a_power == 2 or b_power == 2:
                return "n²×log(n)"
        
        # Caso general: concatenar
        return f"{a}×{b}"
    
    def _get_n_power(self, expr: str) -> Optional[int]:
        """
        Obtiene la potencia de n en una expresión.
        
        Ejemplos:
        - "n" → 1
        - "n²" → 2
        - "n³" → 3
        - "n×n" → 2
        - "5" → None (no es potencia de n)
        """
        expr = expr.strip()
        
        if expr == "n":
            return 1
        elif expr == "n²":
            return 2
        elif expr == "n³":
            return 3
        elif expr == "n⁴":
            return 4
        elif "n^" in expr:
            # Extraer el exponente
            try:
                power = int(expr.split("^")[1])
                return power
            except:
                return None
        elif "n×n×n" in expr:
            return 3
        elif "n×n" in expr:
            return 2
        else:
            return None
    
    def _simplify_expression(self, expr: str) -> str:
        """
        Simplifica una expresión matemática a su forma más simple.
        
        Reglas:
        - Constantes (números) → "1"
        - n-1 → n (términos de orden inferior se ignoran)
        - n+1 → n
        - 2n → n (constantes multiplicativas se ignoran)
        - n/2 → n
        - (n-1)×(n-i) → n² (aproximación)
        """
        expr = str(expr).strip()
        
        # Caso 1: Constantes → "1"
        try:
            val = int(expr)
            return "1"  # Cualquier constante es O(1)
        except:
            pass
        
        # Caso 2: Expresiones con n
        if 'n' in expr:
            # n-1, n-2, etc. → n
            if '-' in expr and not '×' in expr:
                return 'n'
            
            # n+1, n+2, etc. → n
            if '+' in expr and not '×' in expr:
                return 'n'
            
            # 2n, 3n, etc. → n (ignorar constantes)
            if expr[0].isdigit() and 'n' in expr:
                return 'n'
            
            # n/2, n/3, etc. → n
            if '/' in expr and 'n' in expr:
                return 'n'
            
            # (n-i) → n
            if '(' in expr and ')' in expr and '-' in expr:
                return 'n'
        
        # Caso 3: Ya es una forma estándar
        if expr in ['1', 'n', 'n²', 'n³', 'n⁴', 'log(n)', 'n×log(n)', 'n²×log(n)', '2^n', 'n!']:
            return expr
        
        # Caso 4: Multiplicaciones
        if '×' in expr:
            parts = expr.split('×')
            
            # Contar cuántas veces aparece 'n'
            n_count = sum(1 for p in parts if 'n' in p)
            
            if n_count == 0:
                # Solo constantes → "1"
                return '1'
            elif n_count == 1:
                return 'n'
            elif n_count == 2:
                return 'n²'
            elif n_count == 3:
                return 'n³'
            elif n_count >= 4:
                return f'n^{n_count}'
        
        # No se pudo simplificar, retornar como está
        return expr
    
    def _normalize_complexity(self, complexity: str) -> str:
        """
        Normaliza una complejidad a su forma estándar.
        
        Convierte constantes a O(1):
        - O(10) → O(1)
        - O(100) → O(1)
        - O(5) → O(1)
        """
        # Extraer el orden
        order = self._extract_order(complexity)
        
        # Simplificar el orden
        simplified = self._simplify_expression(order)
        
        # Reconstruir con la notación apropiada
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
        """
        Toma el máximo de dos complejidades.
        """
        if self._is_greater_complexity(a.worst_case, b.worst_case):
            return a
        else:
            return b
    
    def _is_greater_complexity(self, a: str, b: str) -> bool:
        """
        Compara dos complejidades: ¿a > b?
        
        Orden: 1 < log(n) < n < n×log(n) < n² < n³ < 2^n < n!
        """
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
        """
        Extrae el orden de una notación O/Ω/Θ.
        
        Ejemplo: "O(n²)" → "n²"
        """
        # Remover O(, Ω(, Θ( y )
        for prefix in ["O(", "Ω(", "Θ("]:
            if complexity.startswith(prefix):
                complexity = complexity[len(prefix):-1]
        
        return complexity
    
    # ========================================================================
    # UTILIDADES
    # ========================================================================
    
    def _expr_to_str(self, expr: ExpressionNode) -> str:
        """Convierte una expresión a string legible"""
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
# FUNCIÓN DE CONVENIENCIA
# ============================================================================

def analyze_complexity(ast: ProgramNode) -> Dict[str, ComplexityResult]:
    """
    Función helper para analizar un AST completo.
    
    Args:
        ast: Programa parseado
        
    Returns:
        Diccionario con resultados por procedimiento
    """
    analyzer = BasicComplexityAnalyzer()
    return analyzer.analyze_program(ast)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    from parser.parser import parse
    
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
    
    print("="*70)
    print("ANÁLISIS DE COMPLEJIDAD - BUBBLE SORT")
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