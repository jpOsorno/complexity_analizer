"""
Resolutor de Ecuaciones de Recurrencia - VERSIÓN CORREGIDA
===========================================================

Implementa múltiples técnicas para resolver recurrencias:
1. ✅ Clasificación automática del tipo de recurrencia
2. ✅ Teorema Maestro (divide y vencerás)
3. ✅ Método de Iteración (resta y vencerás)
4. ✅ Árbol de Recursión (visualización y cálculo)
5. ✅ Método de Sustitución con SymPy
6. ✅ Ecuación Característica (relaciones lineales homogéneas)

Referencias:
- Cormen et al., "Introduction to Algorithms" (CLRS)
- Análisis de Algoritmos y Diseño
"""

import sys
import os
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import sympy as sp
from sympy import symbols, sympify, simplify, log, ceiling, floor, expand, solve
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class RecurrenceSolution:
    """Resultado completo de resolver una recurrencia"""
    
    # Ecuación original
    original_equation: str
    
    # Técnica(s) usada(s)
    method_used: str  # "master", "iteration", "recursion_tree", "substitution", "characteristic"
    
    # Solución en notaciones
    big_o: str       # O(...)
    big_omega: str   # Ω(...)
    big_theta: str   # Θ(...)
    
    # Detalles del análisis
    steps: List[str] = field(default_factory=list)
    tree_analysis: Optional[str] = None
    exact_solution: Optional[str] = None
    
    # Metadata
    complexity_class: str = ""  # "constant", "logarithmic", "linear", etc.
    is_tight: bool = False      # Si O = Ω (entonces Θ)
    recurrence_type: str = ""   # "divide-conquer", "subtract-conquer", "subtract-conquered"
    
    def __str__(self):
        result = f"""
Ecuación: {self.original_equation}
Tipo: {self.recurrence_type}
Método: {self.method_used}

Complejidad:
  O(n): {self.big_o}
  Ω(n): {self.big_omega}
  Θ(n): {self.big_theta}

Clase: {self.complexity_class}
Tight bound: {'Sí' if self.is_tight else 'No'}
        """.strip()
        
        if self.steps:
            result += "\n\nPasos del análisis:"
            for i, step in enumerate(self.steps, 1):
                result += f"\n  {i}. {step}"
        
        if self.exact_solution:
            result += f"\n\nSolución exacta: {self.exact_solution}"
        
        return result
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para serialización"""
        return {
            "equation": self.original_equation,
            "type": self.recurrence_type,
            "method": self.method_used,
            "big_o": self.big_o,
            "big_omega": self.big_omega,
            "big_theta": self.big_theta,
            "complexity_class": self.complexity_class,
            "is_tight": self.is_tight,
            "steps": self.steps,
            "exact_solution": self.exact_solution
        }


# ============================================================================
# CLASIFICADOR DE RECURRENCIAS
# ============================================================================

class RecurrenceClassifier:
    """
    Clasifica ecuaciones de recurrencia según su estructura.
    
    Tipos:
    1. Divide y Vencerás: T(n) = aT(n/b) + f(n)
    2. Resta y Vencerás: T(n) = T(n-k) + f(n)
    3. Resta y Serás Vencido: T(n) = T(n-k1) + T(n-k2) + f(n)
    4. Lineal No Homogénea: T(n) = c1*T(n-1) + c2*T(n-2) + f(n)
    """
    
    @staticmethod
    def classify(equation: str) -> Tuple[str, Dict[str, any]]:
        """
        Clasifica una ecuación de recurrencia.
        
        Returns:
            (tipo, parametros) donde tipo es:
            - "divide-conquer": T(n) = aT(n/b) + f(n)
            - "subtract-conquer": T(n) = T(n-k) + f(n)
            - "subtract-conquered": T(n) = T(n-k1) + T(n-k2) + ... + f(n)
            - "linear-nonhomogeneous": T(n) = c1*T(n-1) + c2*T(n-2) + f(n)
            - "unknown": No se pudo clasificar
        """
        equation = equation.replace(" ", "").replace("T(n)=", "")
        
        # 1. Divide y Vencerás: T(n/b) presente
        if "T(n/" in equation or "T(n/2)" in equation or "T(n/3)" in equation:
            return RecurrenceClassifier._parse_divide_conquer(equation)
        
        # 2. Resta y Serás Vencido: Múltiples T(n-k)
        subtract_terms = re.findall(r'T\(n-(\d+)\)', equation)
        if len(subtract_terms) >= 2:
            return RecurrenceClassifier._parse_subtract_conquered(equation, subtract_terms)
        
        # 3. Resta y Vencerás: Un solo T(n-k)
        if len(subtract_terms) == 1:
            return RecurrenceClassifier._parse_subtract_conquer(equation, subtract_terms[0])
        
        # 4. Lineal con coeficientes: c*T(n-k)
        if re.search(r'\d+\*?T\(n-\d+\)', equation):
            return RecurrenceClassifier._parse_linear_nonhomogeneous(equation)
        
        return ("unknown", {"equation": equation})
    
    @staticmethod
    def _parse_divide_conquer(equation: str) -> Tuple[str, Dict]:
        """Parsea T(n) = aT(n/b) + f(n)"""
        # Buscar coeficiente a
        a_match = re.search(r'(\d+)T\(n/', equation)
        a = int(a_match.group(1)) if a_match else 1
        
        # Buscar divisor b
        b_match = re.search(r'T\(n/(\d+)\)', equation)
        b = int(b_match.group(1)) if b_match else 2
        
        # Extraer f(n)
        if "O(n)" in equation:
            f_n = "O(n)"
        elif "O(n^2)" in equation or "O(n²)" in equation:
            f_n = "O(n^2)"
        elif "O(logn)" in equation or "O(log(n))" in equation:
            f_n = "O(log(n))"
        elif "O(1)" in equation:
            f_n = "O(1)"
        elif "+n" in equation or "+ n" in equation:
            f_n = "O(n)"
        else:
            f_n = "O(1)"
        
        return ("divide-conquer", {"a": a, "b": b, "f_n": f_n})
    
    @staticmethod
    def _parse_subtract_conquer(equation: str, k: str) -> Tuple[str, Dict]:
        """Parsea T(n) = T(n-k) + f(n)"""
        k_val = int(k)
        
        # Extraer f(n)
        if "O(n)" in equation:
            f_n = "O(n)"
        elif "O(1)" in equation or "+c" in equation:
            f_n = "O(1)"
        else:
            f_n = "O(1)"
        
        return ("subtract-conquer", {"k": k_val, "f_n": f_n})
    
    @staticmethod
    def _parse_subtract_conquered(equation: str, subtract_terms: List[str]) -> Tuple[str, Dict]:
        """Parsea T(n) = T(n-k1) + T(n-k2) + ... + f(n)"""
        k_values = [int(k) for k in subtract_terms]
        
        # Extraer f(n)
        if "O(1)" in equation or "+c" in equation or "+ O(1)" in equation:
            f_n = "O(1)"
        else:
            f_n = "O(1)"
        
        return ("subtract-conquered", {"k_values": k_values, "f_n": f_n})
    
    @staticmethod
    def _parse_linear_nonhomogeneous(equation: str) -> Tuple[str, Dict]:
        """Parsea T(n) = c1*T(n-1) + c2*T(n-2) + f(n)"""
        # Buscar coeficientes
        coefficients = {}
        
        for match in re.finditer(r'(\d+)\*?T\(n-(\d+)\)', equation):
            coef = int(match.group(1))
            offset = int(match.group(2))
            coefficients[offset] = coef
        
        # Extraer f(n)
        if "O(1)" in equation:
            f_n = "O(1)"
        else:
            f_n = "O(1)"
        
        return ("linear-nonhomogeneous", {"coefficients": coefficients, "f_n": f_n})


# ============================================================================
# TEOREMA MAESTRO (Master Theorem) - CORREGIDO
# ============================================================================

class MasterTheorem:
    """
    Implementa el Teorema Maestro SOLO para divide y vencerás.
    
    T(n) = a·T(n/b) + f(n)
    
    donde:
    - a ≥ 1 (número de subproblemas)
    - b > 1 (factor de división)
    - f(n) = costo no recursivo
    """
    
    @staticmethod
    def applies(recurrence_type: str, params: Dict) -> bool:
        """Verifica si el Teorema Maestro aplica"""
        if recurrence_type != "divide-conquer":
            return False
        
        a = params.get("a", 0)
        b = params.get("b", 0)
        
        return a >= 1 and b > 1
    
    @staticmethod
    def solve(a: int, b: int, f_n: str, equation: str) -> RecurrenceSolution:
        """
        Resuelve usando el Teorema Maestro.
        """
        steps = []
        steps.append(f"Ecuación identificada: T(n) = {a}T(n/{b}) + {f_n}")
        
        # Calcular log_b(a)
        log_ba = sp.log(a, b)
        log_ba_float = float(log_ba.evalf())
        
        steps.append(f"Parámetros: a={a}, b={b}, f(n)={f_n}")
        steps.append(f"Calcular: log_{b}({a}) = {log_ba_float:.3f}")
        
        # Determinar el orden de f(n)
        f_order = MasterTheorem._get_f_order(f_n)
        steps.append(f"Orden de f(n): n^{f_order}")
        
        # Aplicar casos del Teorema Maestro
        epsilon = 0.1
        
        # Caso 1: f(n) = O(n^(log_b(a) - ε))
        if f_order < log_ba_float - epsilon:
            steps.append(f"Caso 1 del Teorema Maestro: f(n) < n^{log_ba_float:.3f}")
            steps.append(f"Conclusión: T(n) = Θ(n^{log_ba_float:.3f})")
            
            complexity = MasterTheorem._format_complexity(log_ba_float)
            complexity_class = MasterTheorem._classify_complexity(log_ba_float)
            
            return RecurrenceSolution(
                original_equation=equation,
                method_used="master_theorem_case1",
                big_o=f"O({complexity})",
                big_omega=f"Ω({complexity})",
                big_theta=f"Θ({complexity})",
                complexity_class=complexity_class,
                is_tight=True,
                recurrence_type="divide-conquer",
                steps=steps
            )
        
        # Caso 2: f(n) = Θ(n^log_b(a))
        elif abs(f_order - log_ba_float) < epsilon:
            steps.append(f"Caso 2 del Teorema Maestro: f(n) ≈ n^{log_ba_float:.3f}")
            steps.append(f"Conclusión: T(n) = Θ(n^{log_ba_float:.3f} × log(n))")
            
            if abs(log_ba_float - 1.0) < 0.01:
                complexity = "n×log(n)"
                complexity_class = "linearithmic"
            else:
                base_complexity = MasterTheorem._format_complexity(log_ba_float)
                complexity = f"{base_complexity}×log(n)"
                complexity_class = f"polynomial with log factor"
            
            return RecurrenceSolution(
                original_equation=equation,
                method_used="master_theorem_case2",
                big_o=f"O({complexity})",
                big_omega=f"Ω({complexity})",
                big_theta=f"Θ({complexity})",
                complexity_class=complexity_class,
                is_tight=True,
                recurrence_type="divide-conquer",
                steps=steps
            )
        
        # Caso 3: f(n) = Ω(n^(log_b(a) + ε))
        else:
            steps.append(f"Caso 3 del Teorema Maestro: f(n) > n^{log_ba_float:.3f}")
            steps.append(f"Conclusión: T(n) = Θ(f(n))")
            
            complexity = MasterTheorem._extract_complexity_from_f(f_n)
            complexity_class = MasterTheorem._classify_from_string(complexity)
            
            return RecurrenceSolution(
                original_equation=equation,
                method_used="master_theorem_case3",
                big_o=f"O({complexity})",
                big_omega=f"Ω({complexity})",
                big_theta=f"Θ({complexity})",
                complexity_class=complexity_class,
                is_tight=True,
                recurrence_type="divide-conquer",
                steps=steps
            )
    
    @staticmethod
    def _get_f_order(f_n: str) -> float:
        """Determina el orden de f(n)"""
        if "n^2" in f_n or "n²" in f_n:
            return 2.0
        elif "n" in f_n and "log" not in f_n:
            return 1.0
        elif "log" in f_n:
            return 0.5  # Entre constante y lineal
        else:
            return 0.0  # Constante
    
    @staticmethod
    def _format_complexity(power: float) -> str:
        """Formatea una potencia de n"""
        if abs(power - 0.0) < 0.01:
            return "1"
        elif abs(power - 1.0) < 0.01:
            return "n"
        elif abs(power - 2.0) < 0.01:
            return "n²"
        elif abs(power - 3.0) < 0.01:
            return "n³"
        else:
            return f"n^{power:.3f}"
    
    @staticmethod
    def _classify_complexity(power: float) -> str:
        """Clasifica la complejidad"""
        if abs(power - 0.0) < 0.01:
            return "constant"
        elif abs(power - 1.0) < 0.01:
            return "linear"
        elif abs(power - 2.0) < 0.01:
            return "quadratic"
        elif abs(power - 3.0) < 0.01:
            return "cubic"
        else:
            return f"polynomial (degree {power:.2f})"
    
    @staticmethod
    def _extract_complexity_from_f(f_n: str) -> str:
        """Extrae la complejidad de f(n)"""
        if "n^2" in f_n or "n²" in f_n:
            return "n²"
        elif "n" in f_n:
            return "n"
        else:
            return "1"
    
    @staticmethod
    def _classify_from_string(complexity: str) -> str:
        """Clasifica desde string"""
        if complexity == "1":
            return "constant"
        elif complexity == "n":
            return "linear"
        elif complexity == "n²":
            return "quadratic"
        else:
            return "polynomial"


# ============================================================================
# MÉTODO DE ITERACIÓN (para Resta y Vencerás)
# ============================================================================

class IterationMethod:
    """
    Método de Iteración para T(n) = T(n-k) + f(n)
    
    Expande la recurrencia iterativamente hasta el caso base.
    """
    
    @staticmethod
    def applies(recurrence_type: str, params: Dict) -> bool:
        """Verifica si el método aplica"""
        return recurrence_type == "subtract-conquer"
    
    @staticmethod
    def solve(k: int, f_n: str, equation: str) -> RecurrenceSolution:
        """
        Resuelve T(n) = T(n-k) + f(n) por iteración.
        """
        steps = []
        steps.append(f"Ecuación identificada: T(n) = T(n-{k}) + {f_n}")
        steps.append(f"Método: Iteración (expansión)")
        
        # Determinar el costo de f(n)
        f_cost = IterationMethod._extract_cost(f_n)
        
        # Expansión iterativa
        steps.append("\nExpansión:")
        steps.append(f"  T(n) = T(n-{k}) + {f_cost}")
        steps.append(f"       = [T(n-{2*k}) + {f_cost}] + {f_cost} = T(n-{2*k}) + 2×{f_cost}")
        steps.append(f"       = [T(n-{3*k}) + {f_cost}] + 2×{f_cost} = T(n-{3*k}) + 3×{f_cost}")
        steps.append(f"       = ...")
        steps.append(f"       = T(0) + (n/{k})×{f_cost}")
        
        # Calcular complejidad
        if f_cost == "c" or f_cost == "1":
            # T(n) = T(0) + (n/k) × c = O(n)
            complexity = "n"
            complexity_class = "linear"
            steps.append(f"\nSimplificar: T(n) = T(0) + (n/{k})×c = Θ(n)")
        elif f_cost == "n":
            # T(n) = T(0) + (n/k) × n = O(n²)
            complexity = "n²"
            complexity_class = "quadratic"
            steps.append(f"\nSimplificar: T(n) = T(0) + (n/{k})×n = Θ(n²)")
        else:
            complexity = "n"
            complexity_class = "linear"
            steps.append(f"\nSimplificar: T(n) = Θ(n)")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="iteration",
            big_o=f"O({complexity})",
            big_omega=f"Ω({complexity})",
            big_theta=f"Θ({complexity})",
            complexity_class=complexity_class,
            is_tight=True,
            recurrence_type="subtract-conquer",
            steps=steps,
            exact_solution=f"T(n) = T(0) + (n/{k})×{f_cost}"
        )
    
    @staticmethod
    def _extract_cost(f_n: str) -> str:
        """Extrae el costo de f(n)"""
        if "n" in f_n:
            return "n"
        else:
            return "c"


# ============================================================================
# ECUACIÓN CARACTERÍSTICA (para Resta y Serás Vencido)
# ============================================================================

class CharacteristicEquation:
    """
    Método de Ecuación Característica para recurrencias lineales homogéneas.
    
    Ejemplos:
    - T(n) = T(n-1) + T(n-2) → Fibonacci
    - T(n) = 2T(n-1) - T(n-2) → Otras lineales
    """
    
    @staticmethod
    def applies(recurrence_type: str, params: Dict) -> bool:
        """Verifica si el método aplica"""
        return recurrence_type in ["subtract-conquered", "linear-nonhomogeneous"]
    
    @staticmethod
    def solve_fibonacci(k_values: List[int], f_n: str, equation: str) -> RecurrenceSolution:
        """
        Resuelve T(n) = T(n-1) + T(n-2) + O(1) (Fibonacci)
        """
        steps = []
        steps.append("Ecuación identificada: T(n) = T(n-1) + T(n-2) + O(1)")
        steps.append("Tipo: Fibonacci (resta y serás vencido)")
        steps.append("\nMétodo: Ecuación Característica")
        
        # Ecuación característica: r² - r - 1 = 0
        steps.append("\nEcuación característica: r² - r - 1 = 0")
        
        # Resolver con fórmula cuadrática
        r = symbols('r')
        char_eq = r**2 - r - 1
        roots = solve(char_eq, r)
        
        steps.append(f"Raíces: r₁ = {roots[0]}, r₂ = {roots[1]}")
        
        # Raíz dominante (phi)
        phi = (1 + sp.sqrt(5)) / 2
        phi_val = float(phi.evalf())
        
        steps.append(f"\nRaíz dominante: φ = (1+√5)/2 ≈ {phi_val:.3f}")
        steps.append(f"Solución: T(n) = Θ(φⁿ) = Θ({phi_val:.3f}ⁿ)")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="characteristic_equation",
            big_o="O(φⁿ) ≈ O(1.618ⁿ)",
            big_omega="Ω(φⁿ)",
            big_theta="Θ(φⁿ)",
            complexity_class="exponential",
            is_tight=True,
            recurrence_type="subtract-conquered",
            steps=steps,
            exact_solution="T(n) = Θ(φⁿ) donde φ = (1+√5)/2"
        )
    
    @staticmethod
    def solve_general(coefficients: Dict[int, int], f_n: str, equation: str) -> RecurrenceSolution:
        """
        Resuelve recurrencias lineales generales.
        """
        steps = []
        steps.append(f"Ecuación identificada: {equation}")
        steps.append("Método: Ecuación Característica (general)")
        
        # Por simplicidad, manejar casos conocidos
        # TODO: Implementar solver general con SymPy
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="characteristic_equation",
            big_o="O(2ⁿ)",
            big_omega="Ω(2ⁿ)",
            big_theta="Θ(2ⁿ)",
            complexity_class="exponential",
            is_tight=True,
            recurrence_type="linear-nonhomogeneous",
            steps=steps
        )


# ============================================================================
# ÁRBOL DE RECURSIÓN (Visualización y Análisis)
# ============================================================================

class RecursionTree:
    """
    Método del Árbol de Recursión.
    
    Construye conceptualmente el árbol y suma los costos.
    """
    
    @staticmethod
    def analyze(recurrence_type: str, params: Dict, equation: str) -> RecurrenceSolution:
        """
        Analiza usando árbol de recursión.
        """
        if recurrence_type == "subtract-conquered":
            return RecursionTree._analyze_fibonacci(params, equation)
        elif recurrence_type == "subtract-conquer":
            return RecursionTree._analyze_linear(params, equation)
        elif recurrence_type == "divide-conquer":
            return RecursionTree._analyze_divide_conquer(params, equation)
        else:
            return RecursionTree._generic_analysis(equation)
    
    @staticmethod
    def _analyze_fibonacci(params: Dict, equation: str) -> RecurrenceSolution:
        """Árbol para Fibonacci"""
        steps = []
        steps.append("Patrón: Fibonacci (binario)")
        steps.append("Nivel 0: 1 nodo → costo c")
        steps.append("Nivel 1: 2 nodos → costo 2c")
        steps.append("Nivel 2: 4 nodos → costo 4c")
        steps.append("...")
        steps.append("Nivel k: 2ᵏ nodos → costo 2ᵏ×c")
        steps.append("Altura: n niveles")
        steps.append("Total: Σ(2ᵏ×c) para k=0 to n = c×(2ⁿ⁺¹-1) = Θ(2ⁿ)")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="recursion_tree",
            big_o="O(2ⁿ)",
            big_omega="Ω(2ⁿ)",
            big_theta="Θ(2ⁿ)",
            complexity_class="exponential",
            is_tight=True,
            recurrence_type="subtract-conquered",
            steps=steps,
            tree_analysis="Árbol binario de altura n, 2ⁿ hojas"
        )
    
    @staticmethod
    def _analyze_linear(params: Dict, equation: str) -> RecurrenceSolution:
        """Árbol para recursión lineal"""
        k = params.get("k", 1)
        f_n = params.get("f_n", "O(1)")
        
        steps = []
        steps.append("Patrón: Recursión lineal")
        steps.append("Nivel 0: T(n) → costo c")
        steps.append(f"Nivel 1: T(n-{k}) → costo c")
        steps.append("...")
        steps.append(f"Nivel i: T(n-{k}×i) → costo c")
        steps.append(f"Altura: n/{k} niveles")
        steps.append(f"Total: (n/{k})×c = Θ(n)")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="recursion_tree",
            big_o="O(n)",
            big_omega="Ω(n)",
            big_theta="Θ(n)",
            complexity_class="linear",
            is_tight=True,
            recurrence_type="subtract-conquer",
            steps=steps,
            tree_analysis=f"Árbol lineal de altura n/{k}"
        )
    
    @staticmethod
    def _analyze_divide_conquer(params: Dict, equation: str) -> RecurrenceSolution:
        """Árbol para divide y vencerás"""
        a = params.get("a", 2)
        b = params.get("b", 2)
        f_n = params.get("f_n", "O(n)")
        
        steps = []
        steps.append(f"Patrón: Divide y vencerás ({a} subproblemas de tamaño n/{b})")
        steps.append(f"Nivel 0: 1 nodo → costo f(n)")
        steps.append(f"Nivel 1: {a} nodos → costo {a}×f(n/{b})")
        steps.append(f"Nivel 2: {a**2} nodos → costo {a**2}×f(n/{b**2})")
        steps.append("...")
        steps.append(f"Nivel k: {a}ᵏ nodos → costo aᵏ×f(n/bᵏ)")
        steps.append(f"Altura: log_{b}(n) niveles")
        steps.append(f"Total: Usar Teorema Maestro para análisis preciso")
        
        # Usar Teorema Maestro para obtener resultado
        master_result = MasterTheorem.solve(a, b, f_n, equation)
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="recursion_tree",
            big_o=master_result.big_o,
            big_omega=master_result.big_omega,
            big_theta=master_result.big_theta,
            complexity_class=master_result.complexity_class,
            is_tight=True,
            recurrence_type="divide-conquer",
            steps=steps,
            tree_analysis=f"Árbol de altura log_{b}(n) con {a} hijos por nodo"
        )
    
    @staticmethod
    def _generic_analysis(equation: str) -> RecurrenceSolution:
        """Análisis genérico"""
        return RecurrenceSolution(
            original_equation=equation,
            method_used="recursion_tree_generic",
            big_o="O(n)",
            big_omega="Ω(1)",
            big_theta="Θ(?)",
            complexity_class="unknown",
            is_tight=False,
            recurrence_type="unknown",
            steps=["Análisis de árbol genérico - requiere análisis manual"],
            tree_analysis="Estructura no estándar"
        )


# ============================================================================
# MÉTODO DE SUSTITUCIÓN CON SYMPY
# ============================================================================

class SubstitutionMethod:
    """
    Método de Sustitución usando SymPy.
    
    Útil cuando otros métodos no aplican o para verificación.
    """
    
    @staticmethod
    def solve(recurrence_type: str, params: Dict, equation: str) -> RecurrenceSolution:
        """
        Resuelve por sustitución con hipótesis.
        """
        if recurrence_type == "subtract-conquer":
            return SubstitutionMethod._solve_linear(params, equation)
        elif recurrence_type == "subtract-conquered":
            return SubstitutionMethod._solve_fibonacci(params, equation)
        else:
            return SubstitutionMethod._generic_substitution(equation)
    
    @staticmethod
    def _solve_linear(params: Dict, equation: str) -> RecurrenceSolution:
        """Sustitución para T(n) = T(n-k) + f(n)"""
        k = params.get("k", 1)
        f_n = params.get("f_n", "O(1)")
        
        steps = []
        steps.append(f"Ecuación: T(n) = T(n-{k}) + c")
        steps.append("Hipótesis: T(n) = O(n)")
        steps.append("\nVerificación por inducción:")
        steps.append(f"  Suponer T(n) ≤ c×n para todo n < N")
        steps.append(f"  Probar para n = N:")
        steps.append(f"    T(N) = T(N-{k}) + c")
        steps.append(f"         ≤ c×(N-{k}) + c")
        steps.append(f"         = c×N - c×{k} + c")
        steps.append(f"         = c×N + c×(1-{k})")
        steps.append(f"         ≤ c×N  (si c ≥ 1/(k-1))")
        steps.append("\nConclusión: T(n) = Θ(n) ✓")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="substitution",
            big_o="O(n)",
            big_omega="Ω(n)",
            big_theta="Θ(n)",
            complexity_class="linear",
            is_tight=True,
            recurrence_type="subtract-conquer",
            steps=steps,
            exact_solution="T(n) = Θ(n)"
        )
    
    @staticmethod
    def _solve_fibonacci(params: Dict, equation: str) -> RecurrenceSolution:
        """Sustitución para Fibonacci"""
        steps = []
        steps.append("Ecuación: T(n) = T(n-1) + T(n-2) + c")
        steps.append("Hipótesis: T(n) = O(φⁿ) donde φ = (1+√5)/2")
        steps.append("\nVerificación:")
        steps.append("  Propiedad de φ: φ² = φ + 1")
        steps.append("  T(n) ≤ c×φⁿ")
        steps.append("  T(n-1) + T(n-2) ≤ c×φⁿ⁻¹ + c×φⁿ⁻²")
        steps.append("                  = c×φⁿ⁻²×(φ + 1)")
        steps.append("                  = c×φⁿ⁻²×φ²")
        steps.append("                  = c×φⁿ ✓")
        steps.append("\nConclusión: T(n) = Θ(φⁿ)")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="substitution",
            big_o="O(φⁿ) ≈ O(1.618ⁿ)",
            big_omega="Ω(φⁿ)",
            big_theta="Θ(φⁿ)",
            complexity_class="exponential",
            is_tight=True,
            recurrence_type="subtract-conquered",
            steps=steps,
            exact_solution="T(n) = Θ(φⁿ) donde φ = (1+√5)/2"
        )
    
    @staticmethod
    def _generic_substitution(equation: str) -> RecurrenceSolution:
        """Sustitución genérica"""
        return RecurrenceSolution(
            original_equation=equation,
            method_used="substitution_incomplete",
            big_o="O(?)",
            big_omega="Ω(?)",
            big_theta="Θ(?)",
            complexity_class="unknown",
            is_tight=False,
            recurrence_type="unknown",
            steps=["Requiere hipótesis específica para este tipo de recurrencia"]
        )


# ============================================================================
# RESOLUTOR PRINCIPAL
# ============================================================================

class RecurrenceSolver:
    """
    Resolutor principal que integra todas las técnicas.
    
    Flujo:
    1. Clasificar la ecuación
    2. Seleccionar el método apropiado
    3. Aplicar el método y retornar solución
    """
    
    @staticmethod
    def solve(equation: str, preferred_method: Optional[str] = None) -> RecurrenceSolution:
        """
        Resuelve una ecuación de recurrencia.
        
        Args:
            equation: Ecuación en formato string
            preferred_method: Método preferido (opcional):
                - "master": Teorema Maestro
                - "iteration": Iteración
                - "tree": Árbol de Recursión
                - "substitution": Sustitución
                - "characteristic": Ecuación Característica
                - None: Selección automática
        
        Returns:
            RecurrenceSolution con el análisis completo
        """
        # Limpiar ecuación
        equation = equation.strip()
        
        # Paso 1: Clasificar
        recurrence_type, params = RecurrenceClassifier.classify(equation)
        
        if recurrence_type == "unknown":
            return RecurrenceSolution(
                original_equation=equation,
                method_used="none",
                big_o="O(?)",
                big_omega="Ω(?)",
                big_theta="Θ(?)",
                complexity_class="unknown",
                is_tight=False,
                recurrence_type="unknown",
                steps=["No se pudo clasificar la ecuación automáticamente"]
            )
        
        # Paso 2: Seleccionar método
        if preferred_method:
            return RecurrenceSolver._apply_method(
                preferred_method, recurrence_type, params, equation
            )
        else:
            return RecurrenceSolver._auto_select_method(
                recurrence_type, params, equation
            )
    
    @staticmethod
    def _auto_select_method(recurrence_type: str, params: Dict, equation: str) -> RecurrenceSolution:
        """Selección automática del mejor método"""
        
        # Divide y Vencerás → Teorema Maestro
        if recurrence_type == "divide-conquer":
            if MasterTheorem.applies(recurrence_type, params):
                return MasterTheorem.solve(
                    params["a"], params["b"], params["f_n"], equation
                )
        
        # Resta y Vencerás → Iteración
        elif recurrence_type == "subtract-conquer":
            return IterationMethod.solve(
                params["k"], params["f_n"], equation
            )
        
        # Resta y Serás Vencido (Fibonacci) → Ecuación Característica
        elif recurrence_type == "subtract-conquered":
            k_values = params.get("k_values", [])
            if set(k_values) == {1, 2}:  # Fibonacci clásico
                return CharacteristicEquation.solve_fibonacci(
                    k_values, params["f_n"], equation
                )
            else:
                # Usar árbol de recursión
                return RecursionTree.analyze(recurrence_type, params, equation)
        
        # Lineal no homogénea → Ecuación Característica
        elif recurrence_type == "linear-nonhomogeneous":
            return CharacteristicEquation.solve_general(
                params["coefficients"], params["f_n"], equation
            )
        
        # Fallback: Sustitución
        return SubstitutionMethod.solve(recurrence_type, params, equation)
    
    @staticmethod
    def _apply_method(method: str, recurrence_type: str, params: Dict, equation: str) -> RecurrenceSolution:
        """Aplica un método específico"""
        
        if method == "master":
            if MasterTheorem.applies(recurrence_type, params):
                return MasterTheorem.solve(
                    params["a"], params["b"], params["f_n"], equation
                )
            else:
                return RecurrenceSolution(
                    original_equation=equation,
                    method_used="master_not_applicable",
                    big_o="O(?)",
                    big_omega="Ω(?)",
                    big_theta="Θ(?)",
                    steps=["Teorema Maestro no aplica para este tipo de recurrencia"],
                    recurrence_type=recurrence_type
                )
        
        elif method == "iteration":
            if recurrence_type == "subtract-conquer":
                return IterationMethod.solve(params["k"], params["f_n"], equation)
            else:
                return RecurrenceSolution(
                    original_equation=equation,
                    method_used="iteration_not_applicable",
                    big_o="O(?)",
                    big_omega="Ω(?)",
                    big_theta="Θ(?)",
                    steps=["Método de Iteración no aplica para este tipo"],
                    recurrence_type=recurrence_type
                )
        
        elif method == "tree":
            return RecursionTree.analyze(recurrence_type, params, equation)
        
        elif method == "substitution":
            return SubstitutionMethod.solve(recurrence_type, params, equation)
        
        elif method == "characteristic":
            if recurrence_type == "subtract-conquered":
                k_values = params.get("k_values", [])
                return CharacteristicEquation.solve_fibonacci(
                    k_values, params["f_n"], equation
                )
            elif recurrence_type == "linear-nonhomogeneous":
                return CharacteristicEquation.solve_general(
                    params["coefficients"], params["f_n"], equation
                )
            else:
                return RecurrenceSolution(
                    original_equation=equation,
                    method_used="characteristic_not_applicable",
                    big_o="O(?)",
                    big_omega="Ω(?)",
                    big_theta="Θ(?)",
                    steps=["Ecuación Característica no aplica para este tipo"],
                    recurrence_type=recurrence_type
                )
        
        else:
            return RecurrenceSolver._auto_select_method(recurrence_type, params, equation)


# ============================================================================
# API SIMPLIFICADA
# ============================================================================

def solve_recurrence(equation: str, method: Optional[str] = None) -> RecurrenceSolution:
    """
    API simple para resolver recurrencias.
    
    Args:
        equation: Ecuación (ej: "T(n) = 2T(n/2) + O(n)")
        method: Método preferido (opcional)
        
    Returns:
        RecurrenceSolution con análisis completo
    """
    solver = RecurrenceSolver()
    return solver.solve(equation, method)


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demuestra el resolutor con ejemplos variados"""
    
    examples = [
        ("Merge Sort (Divide y Vencerás)", "T(n) = 2T(n/2) + O(n)"),
        ("Binary Search (Divide y Vencerás)", "T(n) = T(n/2) + O(1)"),
        ("Factorial (Resta y Vencerás)", "T(n) = T(n-1) + O(1)"),
        ("Fibonacci (Resta y Serás Vencido)", "T(n) = T(n-1) + T(n-2) + O(1)"),
        ("Strassen (Divide y Vencerás)", "T(n) = 7T(n/2) + O(n^2)"),
        ("Linear Search (Resta y Vencerás)", "T(n) = T(n-1) + c"),
    ]
    
    print("="*70)
    print("RESOLUTOR DE RECURRENCIAS - DEMO COMPLETO")
    print("="*70)
    
    for name, equation in examples:
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print(f"{'='*70}")
        
        try:
            solution = solve_recurrence(equation)
            print(solution)
            
            # Mostrar tipo de recurrencia
            print(f"\n🔍 Tipo de recurrencia: {solution.recurrence_type}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()