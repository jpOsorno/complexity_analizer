"""
Analizador de Recursión - VERSIÓN MEJORADA CON MÚLTIPLES CASOS
==============================================================

Mejoras:
1. Genera ecuaciones para peor caso, mejor caso y caso promedio
2. Analiza condicionales para detectar diferentes caminos de ejecución
3. Identifica early exits y casos base
4. Calcula complejidad promedio cuando es posible
"""

import sys
import os
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from syntax_tree.nodes import *
from analyzer.visitor import ASTVisitor


# ============================================================================
# ESTRUCTURAS DE DATOS MEJORADAS
# ============================================================================

@dataclass
class RecursiveCall:
    """Representa una llamada recursiva"""
    function_name: str
    arguments: List[ExpressionNode]
    depth_reduction: Optional[str] = None
    location: str = ""
    path_condition: Optional[str] = None  # NUEVO: condición del camino
    in_return: bool = False
    return_group: Optional[int] = None


@dataclass
class RecurrenceEquation:
    """
    Representa ecuaciones de recurrencia para diferentes casos.
    
    MEJORADO: Ahora contiene worst_case, best_case y average_case
    """
    function_name: str
    parameter: str
    
    # Casos base
    base_case_condition: str
    base_case_cost: str
    
    # NUEVO: Ecuaciones para cada caso
    worst_case_equation: str = ""
    best_case_equation: str = ""
    average_case_equation: str = ""
    
    # Información adicional
    recursive_calls: List[RecursiveCall] = field(default_factory=list)
    non_recursive_cost: str = "O(1)"
    recursion_type: str = ""
    
    # NUEVO: Explicaciones
    worst_case_explanation: str = ""
    best_case_explanation: str = ""
    average_case_explanation: str = ""
    
    def __post_init__(self):
        if not self.worst_case_equation:
            self._generate_equations()
        if not self.recursion_type:
            self.recursion_type = self._classify_recursion()
    
    def _generate_equations(self):
        """
        Genera ecuaciones para peor, mejor y caso promedio.
        
        ESTRATEGIA:
        - Peor caso: máxima profundidad de recursión
        - Mejor caso: mínima profundidad (early exit si existe)
        - Promedio: depende del tipo de recursión
        """
        if not self.recursive_calls:
            # No hay recursión - todas iguales
            self.worst_case_equation = f"T({self.parameter}) = {self.base_case_cost}"
            self.best_case_equation = self.worst_case_equation
            self.average_case_equation = self.worst_case_equation
            return
        
        # PEOR CASO: Analizar si las llamadas recursivas son simultáneas o
        # mutuamente excluyentes (p. ej. ramas if/else). Si son exclusivas,
        # la contribución es el máximo en vez de la suma.
        candidate_calls = []
        for call in self.recursive_calls:
            dr = call.depth_reduction
            if not dr:
                continue
            reduction = dr.replace(" ", "").strip("()")

            # Normalizar patrones reconocidos
            if '/2' in reduction or 'n/2' in reduction or 'floor' in reduction:
                term = 'T(n/2)'
            elif 'n-1' in reduction or '-1' in reduction:
                term = 'T(n-1)'
            elif self.parameter and self.parameter in reduction:
                term = f"T({reduction})"
            elif 'n' in reduction:
                term = f"T({reduction})"
            else:
                # No reduce el parámetro principal -> ignorar para la suma
                term = None

            candidate_calls.append((call, term))

        # Si no hay candidatos válidos, usar T(?) como fallback
        if not any(term for _, term in candidate_calls):
            worst_recursive_part = 'T(?)'
            self.worst_case_explanation = 'Peor caso: llamadas recursivas no determinables'
        else:
            # Detectar si las llamadas están en distintas condiciones (mutuamente
            # excluyentes). Si hay al menos dos llamadas con path_condition
            # diferentes y no vacías, tratamos las llamadas como exclusivas.
            path_conditions = set(c.path_condition for c, _ in candidate_calls if c.path_condition)
            exclusive = len(path_conditions) > 1

            terms = [t for _, t in candidate_calls if t]
            # Separar llamadas que están dentro de `return` (podrían ser
            # exclusivas si vienen de distintos `return`s) de las que no lo están.
            from collections import defaultdict
            in_return_groups: Dict[Optional[int], List[str]] = defaultdict(list)
            non_return_terms: List[str] = []
            for c, t in candidate_calls:
                if not t:
                    continue
                if getattr(c, 'in_return', False):
                    grp = getattr(c, 'return_group', None)
                    in_return_groups[grp].append(t)
                else:
                    non_return_terms.append(t)

            if exclusive:
                # Elegir la "mayor" reducción detectada como representante.
                # Preferir n/2 > n-1 > otros
                preferred = None
                for pref in ('T(n/2)', 'T(n-1)'):
                    if pref in terms:
                        preferred = pref
                        break
                if not preferred:
                    preferred = terms[0]
                worst_recursive_part = preferred
                self.worst_case_explanation = 'Peor caso: ramas mutuamente excluyentes, tomar la más costosa'
            else:
                # No exclusivas -> sumar contribuciones no-return y además
                # considerar las llamadas dentro de return como exclusivas entre sí.
                from collections import Counter, defaultdict
                parts = []

                if non_return_terms:
                    # Normalizar términos para que diferencias en espacios u
                    # otros formatos equivalentes se cuenten juntos.
                    # Usamos una clave canónica sin espacios para contar,
                    # y guardamos una representación legible para salida.
                    canon_map: Dict[str, str] = {}
                    canon_keys = []
                    for t in non_return_terms:
                        key = "".join(t.split())
                        canon_keys.append(key)
                        # Guardar la primera representación legible encontrada
                        if key not in canon_map:
                            canon_map[key] = t

                    cnt = Counter(canon_keys)
                    for key, c in cnt.items():
                        term = canon_map.get(key, key)
                        if c > 1 and term.startswith('T('):
                            parts.append(f"{c}{term}")
                        else:
                            # Si no es una T(...) o no queremos multiplicar,
                            # agregar tantas ocurrencias como haya (para suma)
                            if c > 1:
                                parts.extend([term] * c)
                            else:
                                parts.append(term)

                # Procesar grupos de llamadas in-return
                if in_return_groups:
                    # Si sólo hay UN grupo, todas las llamadas vienen del mismo
                    # `return` (por ejemplo, `return f(n-1) + f(n-2)`), por lo que
                    # deben sumarse.
                    if len(in_return_groups) == 1:
                        group_terms = next(iter(in_return_groups.values()))
                        # Normalizar y contar dentro del grupo
                        canon_map_g: Dict[str, str] = {}
                        keys_g = []
                        for t in group_terms:
                            key = "".join(t.split())
                            keys_g.append(key)
                            if key not in canon_map_g:
                                canon_map_g[key] = t
                        cnt_g = Counter(keys_g)
                        for key, c in cnt_g.items():
                            term = canon_map_g.get(key, key)
                            if c > 1 and term.startswith('T('):
                                parts.append(f"{c}{term}")
                            else:
                                if c > 1:
                                    parts.extend([term] * c)
                                else:
                                    parts.append(term)
                    else:
                        # Hay múltiples grupos (llamadas en distintos `return`),
                        # tratamos los grupos como alternativas mutuamente
                        # excluyentes y elegimos la representante más costosa
                        representatives = []
                        for grp_terms in in_return_groups.values():
                            preferred = None
                            for pref in ('T(n/2)', 'T(n-1)'):
                                if pref in grp_terms:
                                    preferred = pref
                                    break
                            if not preferred:
                                preferred = grp_terms[0]
                            representatives.append(preferred)

                        preferred_in = None
                        for pref in ('T(n/2)', 'T(n-1)'):
                            if pref in representatives:
                                preferred_in = pref
                                break
                        if not preferred_in:
                            preferred_in = representatives[0]
                        parts.append(preferred_in)

                # Compactar términos recursivos idénticos (p.ej. T(n/2) + T(n/2) -> 2T(n/2))
                import re
                from collections import Counter

                # Separar términos T(...) y otros
                t_counter = Counter()
                t_repr: Dict[str, str] = {}
                others: List[str] = []

                for p in parts:
                    if not p or p.strip() == '':
                        continue
                    # Detectar forma 'kT(...)' o 'T(...)'
                    m = re.match(r'^\s*(\d+)\s*(T\(.*\))\s*$', p)
                    if m:
                        k = int(m.group(1))
                        base = m.group(2)
                        key = ''.join(base.split())
                        t_counter[key] += k
                        t_repr[key] = base
                    elif p.strip().startswith('T('):
                        base = p.strip()
                        key = ''.join(base.split())
                        t_counter[key] += 1
                        t_repr[key] = base
                    else:
                        others.append(p)

                final_parts: List[str] = []
                # Añadir términos recursivos compactados
                for key, cnt in t_counter.items():
                    base = t_repr.get(key, key)
                    if cnt == 1:
                        final_parts.append(base)
                    else:
                        final_parts.append(f"{cnt}{base}")

                # Mantener el orden relativo: primero recursivos compactados, luego otros
                final_parts.extend(others)

                worst_recursive_part = ' + '.join(final_parts) if final_parts else 'T(?)'
                self.worst_case_explanation = 'Peor caso: combinación de llamadas (sumadas y exclusivas)'

        self.worst_case_equation = f"T({self.parameter}) = {worst_recursive_part} + {self.non_recursive_cost}"
        # Pasada final: combinar términos recursivos idénticos en la cadena
        # resultante (ej. 'T(n/2) + T(n/2) + n' -> '2T(n/2) + n'). Esto actúa
        # como respaldo si otras heurísticas fallan.
        try:
            import re
            from collections import OrderedDict

            parts = [p.strip() for p in worst_recursive_part.split('+') if p.strip()]
            t_counts = {}
            t_order = []
            others = []

            for p in parts:
                # Detectar formas como 'kT(...)' o 'T(...)'
                m = re.match(r'^\s*(?:(\d+)\s*)?(T\(.+\))\s*$', p)
                if m:
                    k = int(m.group(1)) if m.group(1) else 1
                    base = m.group(2)
                    key = ''.join(base.split())
                    if key not in t_counts:
                        t_counts[key] = 0
                        t_order.append(key)
                    t_counts[key] += k
                    # conservar una representación legible
                    if key not in t_order:
                        t_order.append(key)
                else:
                    others.append(p)

            new_parts = []
            for key in t_order:
                cnt = t_counts.get(key, 0)
                if cnt <= 0:
                    continue
                # obtener base readable (reconstruir T(...) desde key)
                # key es sin espacios, buscamos en parts
                readable = None
                for p in parts:
                    if ''.join(p.split()).endswith(key):
                        # extraer la parte T(...) si está presente
                        m2 = re.search(r'(T\(.+\))', p)
                        if m2:
                            readable = m2.group(1)
                            break
                if not readable:
                    # fallback
                    readable = key
                if cnt == 1:
                    new_parts.append(readable)
                else:
                    new_parts.append(f"{cnt}{readable}")

            # añadir otros (manteniendo el orden aproximado)
            new_parts.extend(others)

            if new_parts:
                worst_recursive_part = ' + '.join(new_parts)
                self.worst_case_equation = f"T({self.parameter}) = {worst_recursive_part} + {self.non_recursive_cost}"
        except Exception:
            # No bloquear si algo falla en esta pasada final
            pass
        
        # MEJOR CASO: Detección de early exit
        has_early_exit = self._has_early_exit_pattern()
        
        if has_early_exit:
            # Mejor caso: llega al caso base inmediatamente
            self.best_case_equation = f"T({self.parameter}) = {self.base_case_cost}"
            self.best_case_explanation = "Mejor caso: se alcanza el caso base directamente (early exit)"
        else:
            # Sin early exit, mejor caso = peor caso
            self.best_case_equation = self.worst_case_equation
            self.best_case_explanation = "Mejor caso igual al peor caso (sin early exit)"
        
        # CASO PROMEDIO: Depende del tipo de recursión
        self.average_case_equation = self._calculate_average_case()
        self.average_case_explanation = self._explain_average_case()
    
    def _has_early_exit_pattern(self) -> bool:
        """
        Detecta si hay patrón de early exit.
        
        Patrón común: if (condición) then return caso_base
        """
        # Si hay condición de caso base clara, hay early exit
        if self.base_case_condition and self.base_case_condition != "unknown":
            return True
        return False
    
    def _calculate_average_case(self) -> str:
        """
        Calcula ecuación de caso promedio según tipo de recursión.
        
        CASOS:
        1. Lineal (n-1): Promedio = Peor caso
        2. Divide and Conquer (n/2): Promedio = Peor caso
        3. Binaria (n-1, n-2): Promedio ≈ Peor caso
        4. Con early exit: Entre mejor y peor caso
        """
        recursion_type = self._classify_recursion()
        
        if recursion_type in ["linear", "divide-and-conquer", "simple"]:
            # Promedio = peor caso para estos tipos
            return self.worst_case_equation
        
        elif recursion_type == "binary":
            # Fibonacci: promedio = peor caso (exponencial)
            return self.worst_case_equation
        
        elif self._has_early_exit_pattern():
            # Con early exit: promedio está entre mejor y peor
            # Usamos la ecuación del peor caso pero indicamos que puede ser mejor
            return self.worst_case_equation.replace("T(", "T_avg(")
        
        else:
            # Por defecto: promedio = peor caso
            return self.worst_case_equation
    
    def _explain_average_case(self) -> str:
        """Genera explicación del caso promedio"""
        recursion_type = self._classify_recursion()
        
        if self._has_early_exit_pattern():
            return "Caso promedio: entre el mejor y peor caso, depende de la entrada"
        elif recursion_type in ["linear", "divide-and-conquer"]:
            return "Caso promedio igual al peor caso (comportamiento determinista)"
        elif recursion_type == "binary":
            return "Caso promedio similar al peor caso (exponencial)"
        else:
            return "Caso promedio aproximado al peor caso"
    
    def _classify_recursion(self) -> str:
        """Clasifica el tipo de recursión"""
        num_calls = len(self.recursive_calls)
        
        if num_calls == 0:
            return "non-recursive"
        elif num_calls == 1:
            call = self.recursive_calls[0]
            if call.depth_reduction:
                reduction = call.depth_reduction.replace(" ", "")
                if "n-1" in reduction or "-1" in reduction:
                    return "linear"
                elif "n/2" in reduction or "/2" in reduction:
                    return "divide-and-conquer"
            return "simple"
        elif num_calls == 2:
            # Si ambas llamadas reducen a n/2, es divide-and-conquer (ej. MergeSort)
            drs = [c.depth_reduction or '' for c in self.recursive_calls]
            if all(d and ('/2' in d or 'n/2' in d or d == 'n/2') for d in drs):
                return 'divide-and-conquer'
            return "binary"
        else:
            return "multiple"
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para serialización"""
        return {
            "function_name": self.function_name,
            "parameter": self.parameter,
            "recursion_type": self.recursion_type,
            "base_case": {
                "condition": self.base_case_condition,
                "cost": self.base_case_cost
            },
            "worst_case": {
                "equation": self.worst_case_equation,
                "explanation": self.worst_case_explanation
            },
            "best_case": {
                "equation": self.best_case_equation,
                "explanation": self.best_case_explanation
            },
            "average_case": {
                "equation": self.average_case_equation,
                "explanation": self.average_case_explanation
            },
            "num_recursive_calls": len(self.recursive_calls)
        }


@dataclass
class RecursionAnalysisResult:
    """Resultado del análisis de recursión con múltiples casos"""
    procedure_name: str
    is_recursive: bool
    recurrence_equation: Optional[RecurrenceEquation] = None
    base_cases: List[str] = field(default_factory=list)
    recursive_cases: List[str] = field(default_factory=list)
    max_recursion_depth: Optional[int] = None
    calls_count: int = 0
    
    def to_recurrence_worst(self) -> Optional[str]:
        """Retorna ecuación del PEOR CASO"""
        if not self.is_recursive or not self.recurrence_equation:
            return None
        return self.recurrence_equation.worst_case_equation
    
    def to_recurrence_best(self) -> Optional[str]:
        """Retorna ecuación del MEJOR CASO"""
        if not self.is_recursive or not self.recurrence_equation:
            return None
        return self.recurrence_equation.best_case_equation
    
    def to_recurrence_average(self) -> Optional[str]:
        """Retorna ecuación del CASO PROMEDIO"""
        if not self.is_recursive or not self.recurrence_equation:
            return None
        return self.recurrence_equation.average_case_equation
    
    def to_recurrence(self) -> Optional[str]:
        """Retorna ecuación del peor caso (por compatibilidad)"""
        return self.to_recurrence_worst()
    
    def get_all_equations(self) -> Optional[Dict[str, str]]:
        """
        Retorna todas las ecuaciones.
        
        Returns:
            Dict con keys: 'worst', 'best', 'average'
        """
        if not self.is_recursive or not self.recurrence_equation:
            return None
        
        return {
            'worst': self.recurrence_equation.worst_case_equation,
            'best': self.recurrence_equation.best_case_equation,
            'average': self.recurrence_equation.average_case_equation
        }


# ============================================================================
# VISITOR MEJORADO
# ============================================================================

class RecursionAnalyzerVisitor(ASTVisitor):
    """Visitor que analiza recursión con detección de múltiples casos"""
    
    def __init__(self, procedure_name: str):
        self.procedure_name = procedure_name
        self.current_procedure = procedure_name
        self.recursive_calls: List[RecursiveCall] = []
        self.base_cases: List[Tuple[str, str]] = []
        self.in_base_case = False
        self.current_condition = None
        self.parameter_name = None
        self.array_param_name = None
        # Mapa para variables locales (e.g., mid -> floor((left+right)/2))
        self.local_var_map: Dict[str, str] = {}
        self.has_merge = False
        
        # NUEVO: Tracking de caminos
        self.current_path_conditions: List[str] = []
        self.has_early_exit = False
        self.current_in_return = False
        # Tracking de grupos de return para distinguir llamadas dentro
        # del mismo `return` (sumables) de llamadas en distintos returns
        # (mutuamente excluyentes)
        self.current_return_group: Optional[int] = None
        self.return_group_counter: int = 0
    
    def visit_program(self, node: ProgramNode):
        for proc in node.procedures:
            if proc.name == self.procedure_name:
                return proc.accept(self)
        return None
    
    def visit_procedure(self, node: ProcedureNode):
        # Identificar parámetro principal
        if node.parameters:
            first_param = node.parameters[0]
            if isinstance(first_param, SimpleParamNode):
                self.parameter_name = first_param.name
            elif isinstance(first_param, ArrayParamNode):
                # Guardar nombre del array para poder ignorarlo al analizar argumentos
                try:
                    self.array_param_name = first_param.name
                except Exception:
                    self.array_param_name = None
                if len(node.parameters) > 1 and isinstance(node.parameters[1], SimpleParamNode):
                    self.parameter_name = node.parameters[1].name
        
        node.body.accept(self)
        return self._build_result()
    
    def visit_if(self, node: IfNode):
        """
        Analiza IF con tracking de caminos.
        
        MEJORA: Detecta si el THEN es caso base (early exit)
        """
        old_condition = self.current_condition
        condition_str = self._condition_to_string(node.condition)
        self.current_condition = condition_str
        
        # Agregar condición al camino
        self.current_path_conditions.append(condition_str)
        
        old_base_flag = self.in_base_case
        
        # Verificar si THEN es caso base (no tiene llamadas recursivas)
        has_recursive_call_in_then = self._has_recursive_call(node.then_block)
        
        if not has_recursive_call_in_then:
            # THEN es caso base
            self.in_base_case = True
            self.has_early_exit = True
            cost = self._estimate_cost(node.then_block)
            self.base_cases.append((condition_str, cost))
        
        # Visitar THEN (IMPORTANTE: siempre visitar, puede haber IFs anidados)
        node.then_block.accept(self)
        self.in_base_case = old_base_flag
        
        # Limpiar condición del THEN antes de procesar ELSE
        self.current_path_conditions.pop()
        
        # Visitar ELSE (caso recursivo)
        if node.else_block:
            # Agregar condición negada para el ELSE
            negated_condition = f"not ({condition_str})"
            self.current_path_conditions.append(negated_condition)
            
            self.in_base_case = False
            node.else_block.accept(self)
            
            # Limpiar condición del ELSE
            self.current_path_conditions.pop()
        
        self.current_condition = old_condition
    
    def visit_for(self, node: ForNode):
        node.body.accept(self)
    
    def visit_while(self, node: WhileNode):
        node.body.accept(self)
    
    def visit_repeat(self, node: RepeatNode):
        node.body.accept(self)
    
    def visit_function_call(self, node: FunctionCallNode):
        if node.name == self.procedure_name:
            call = self._analyze_recursive_call(node)
            # Agregar condición del camino
            if self.current_path_conditions:
                call.path_condition = " AND ".join(self.current_path_conditions)
            # Indicar si la llamada está dentro de un return
            call.in_return = self.current_in_return
            # Asociar al grupo de return actual (si existe)
            call.return_group = self.current_return_group
            self.recursive_calls.append(call)
        # Detectar llamadas a Merge (operación lineal sobre la partición)
        if node.name and node.name.lower() == 'merge':
            self.has_merge = True

        for arg in node.arguments:
            arg.accept(self)
    
    def visit_call_statement(self, node: CallStatementNode):
        if node.name == self.procedure_name:
            call = self._analyze_recursive_call_statement(node)
            if self.current_path_conditions:
                call.path_condition = " AND ".join(self.current_path_conditions)
            call.in_return = self.current_in_return
            # Asociar al grupo de return actual (aunque statements raramente
            # aparecen dentro de un return, lo registramos por completitud)
            call.return_group = self.current_return_group
            self.recursive_calls.append(call)
        # Detectar llamadas a Merge como statement
        if node.name and node.name.lower() == 'merge':
            self.has_merge = True

        for arg in node.arguments:
            arg.accept(self)
    
    def visit_return(self, node: ReturnNode):
        # Marcar contexto de return para que llamadas recursivas dentro se
        # identifiquen como 'in_return'. Esto ayuda a detectar calls mutuamente
        # excluyentes (ej. Binary Search usa return en ramas).
        old_in_return = self.current_in_return
        old_return_group = self.current_return_group
        # Nuevo id de grupo para las llamadas dentro de este `return`
        self.return_group_counter += 1
        self.current_return_group = self.return_group_counter
        self.current_in_return = True
        if node.value:
            node.value.accept(self)
        self.current_in_return = old_in_return
        # Restaurar el grupo previo
        self.current_return_group = old_return_group
    
    def visit_assignment(self, node: AssignmentNode):
        # Visitar el valor primero para procesar subexpresiones
        node.value.accept(self)

        # Registrar asignaciones simples a variables locales para inferencia
        # (ej. mid ← floor((left + right) / 2))
        target = node.target
        try:
            if isinstance(target, VariableLValueNode):
                varname = target.name
                val_str = self._expr_to_string(node.value).replace(" ", "")
                if val_str and val_str != "?":
                    self.local_var_map[varname] = val_str
        except Exception:
            pass
    
    # ========================================================================
    # MÉTODOS AUXILIARES
    # ========================================================================
    
    def _has_recursive_call(self, block: BlockNode) -> bool:
        """
        Verifica si un bloque contiene llamadas recursivas.
        
        IMPORTANTE: Debe recorrer TODOS los nodos, incluyendo IFs anidados
        """
        class CallFinder(ASTVisitor):
            def __init__(self, target_name):
                self.target_name = target_name
                self.found = False
            
            def visit_program(self, node): 
                pass
            
            def visit_procedure(self, node): 
                pass
            
            def visit_for(self, node): 
                if not self.found:
                    node.body.accept(self)
            
            def visit_while(self, node):
                if not self.found:
                    node.condition.accept(self)
                    node.body.accept(self)
            
            def visit_repeat(self, node):
                if not self.found:
                    node.body.accept(self)
                    node.condition.accept(self)
            
            def visit_if(self, node):
                if not self.found:
                    node.condition.accept(self)
                    node.then_block.accept(self)
                    if node.else_block:
                        node.else_block.accept(self)
            
            def visit_assignment(self, node):
                if not self.found:
                    node.value.accept(self)
            
            def visit_return(self, node):
                if not self.found and node.value:
                    node.value.accept(self)
            
            def visit_function_call(self, node):
                if node.name == self.target_name:
                    self.found = True
                else:
                    # Seguir buscando en los argumentos
                    for arg in node.arguments:
                        if not self.found:
                            arg.accept(self)
            
            def visit_call_statement(self, node):
                if node.name == self.target_name:
                    self.found = True
                else:
                    # Seguir buscando en los argumentos
                    for arg in node.arguments:
                        if not self.found:
                            arg.accept(self)
            
            def visit_binary_op(self, node):
                if not self.found:
                    node.left.accept(self)
                    node.right.accept(self)
            
            def visit_unary_op(self, node):
                if not self.found:
                    node.operand.accept(self)
            
            # Para los nodos terminales, no hacer nada
            def visit_number(self, node): pass
            def visit_string(self, node): pass
            def visit_boolean(self, node): pass
            def visit_null(self, node): pass
            def visit_identifier(self, node): pass
        
        finder = CallFinder(self.procedure_name)
        block.accept(finder)
        return finder.found
    
    def _condition_to_string(self, condition: ExpressionNode) -> str:
        """Convierte condición a string normalizado"""
        result = self._expr_to_string(condition)
        return " ".join(result.split())
    
    def _expr_to_string(self, expr: ExpressionNode) -> str:
        """Convierte expresión a string"""
        if isinstance(expr, NumberNode):
            return str(expr.value)
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._expr_to_string(expr.left)
            right = self._expr_to_string(expr.right)
            return f"{left}{expr.op}{right}"
        elif isinstance(expr, UnaryOpNode):
            operand = self._expr_to_string(expr.operand)
            return f"{expr.op}{operand}"
        elif isinstance(expr, BooleanNode):
            return str(expr.value)
        # Añadir soporte para nodos adicionales usados en la gramática
        elif isinstance(expr, ArrayAccessNode):
            # A[ind1][ind2]... -> representamos índices entre corchetes
            indices = ",".join(self._expr_to_string(i) for i in expr.indices)
            return f"{expr.name}[{indices}]"
        elif isinstance(expr, FunctionCallNode):
            args = ",".join(self._expr_to_string(a) for a in expr.arguments)
            return f"call {expr.name}({args})"
        elif hasattr(expr, 'expression') and type(expr).__name__ in ('FloorNode', 'CeilingNode'):
            # Floor/Ceiling wrappers
            inner = self._expr_to_string(getattr(expr, 'expression'))
            if type(expr).__name__ == 'FloorNode':
                return f"floor({inner})"
            else:
                return f"ceiling({inner})"
        elif hasattr(expr, 'array_name') and type(expr).__name__ == 'LengthNode':
            return f"length({expr.array_name})"
        else:
            return "?"
    
    def _analyze_recursive_call(self, node: FunctionCallNode) -> RecursiveCall:
        """Analiza llamada recursiva"""
        depth_reduction = None

        # Buscar en todos los argumentos la expresión que reduzca el parámetro
        if node.arguments:
            # Priorizar argumentos que contengan el nombre del parámetro
            for arg in node.arguments:
                arg_str = self._expr_to_string(arg).replace(" ", "")
                if self.parameter_name and self.parameter_name in arg_str:
                    depth_reduction = arg_str
                    break

            # Si el argumento contiene una variable local (ej. mid, q) mapeada,
            # intentar resolverla a una reducción (ej. n/2)
            if depth_reduction is None and self.local_var_map:
                for varname, mapped in self.local_var_map.items():
                    for arg in node.arguments:
                        arg_str = self._expr_to_string(arg).replace(" ", "")
                        if varname in arg_str:
                            # Heurística: si la asignación contiene '/2' o 'floor',
                            # consideramos una reducción a la mitad
                            if ('/2' in mapped) or ('floor' in mapped) or ('(left+right)/2' in mapped):
                                depth_reduction = 'n/2'
                            else:
                                depth_reduction = mapped
                            break
                    if depth_reduction:
                        break

            # Heurística adicional: si el primer parámetro es un array y los
            # argumentos contienen p/q/r o expresiones con q, y q está mapeado
            # como floor((p+r)/2), entonces asumimos reducción a la mitad.
            if depth_reduction is None and self.array_param_name and self.local_var_map:
                for varname, mapped in self.local_var_map.items():
                    if ('/2' in mapped) or ('floor' in mapped) or ('(p+r)/2' in mapped) or ('(left+right)/2' in mapped):
                        # buscar q o varname en argumentos
                        for arg in node.arguments:
                            arg_str = self._expr_to_string(arg).replace(" ", "")
                            if varname in arg_str or 'q' in arg_str:
                                depth_reduction = 'n/2'
                                break
                        if depth_reduction:
                            break

            # Si no se encontró por nombre, elegir un argumento heurístico
            if depth_reduction is None:
                for arg in node.arguments:
                    arg_str = self._expr_to_string(arg).replace(" ", "")
                    # Ignorar el argumento que es el array cuando procedente
                    if self.array_param_name and arg_str == self.array_param_name:
                        continue
                    # Ignorar literales o identificadores simples que igualan al array
                    if arg_str == "" or arg_str is None:
                        continue
                    # Tomar el primer argumento razonable como fallback
                    depth_reduction = arg_str
                    break

        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location="expression"
        )
    
    def _analyze_recursive_call_statement(self, node: CallStatementNode) -> RecursiveCall:
        """Analiza llamada recursiva como statement"""
        depth_reduction = None

        if node.arguments:
            for arg in node.arguments:
                arg_str = self._expr_to_string(arg).replace(" ", "")
                if self.parameter_name and self.parameter_name in arg_str:
                    depth_reduction = arg_str
                    break

            # Intentar resolver usando variables locales mapeadas
            if depth_reduction is None and self.local_var_map:
                for varname, mapped in self.local_var_map.items():
                    for arg in node.arguments:
                        arg_str = self._expr_to_string(arg).replace(" ", "")
                        if varname in arg_str:
                            if ('/2' in mapped) or ('floor' in mapped) or ('(left+right)/2' in mapped):
                                depth_reduction = 'n/2'
                            else:
                                depth_reduction = mapped
                            break
                    if depth_reduction:
                        break

            # Igual heurística adicional para llamadas como statements
            if depth_reduction is None and self.array_param_name and self.local_var_map:
                for varname, mapped in self.local_var_map.items():
                    if ('/2' in mapped) or ('floor' in mapped) or ('(p+r)/2' in mapped) or ('(left+right)/2' in mapped):
                        for arg in node.arguments:
                            arg_str = self._expr_to_string(arg).replace(" ", "")
                            if varname in arg_str or 'q' in arg_str:
                                depth_reduction = 'n/2'
                                break
                        if depth_reduction:
                            break

            # Resolver por variables locales mapeadas en llamadas como statements
            if depth_reduction is None and self.local_var_map:
                for varname, mapped in self.local_var_map.items():
                    for arg in node.arguments:
                        arg_str = self._expr_to_string(arg).replace(" ", "")
                        if varname in arg_str:
                            if ('/2' in mapped) or ('floor' in mapped) or ('(left+right)/2' in mapped):
                                depth_reduction = 'n/2'
                            else:
                                depth_reduction = mapped
                            break
                    if depth_reduction:
                        break

            if depth_reduction is None:
                for arg in node.arguments:
                    arg_str = self._expr_to_string(arg).replace(" ", "")
                    if self.array_param_name and arg_str == self.array_param_name:
                        continue
                    depth_reduction = arg_str
                    break

        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location="statement"
        )
    
    def _estimate_cost(self, block: BlockNode) -> str:
        """Estima costo del caso base"""
        return "O(1)"
    
    def _build_result(self) -> RecursionAnalysisResult:
        """Construye resultado final con ecuaciones para todos los casos"""
        is_recursive = len(self.recursive_calls) > 0
        
        if not is_recursive:
            return RecursionAnalysisResult(
                procedure_name=self.procedure_name,
                is_recursive=False
            )
        
        # Determinar caso base
        if self.base_cases:
            base_condition, base_cost = self.base_cases[0]
        else:
            base_condition = "unknown"
            base_cost = "O(1)"
        
        # Crear ecuación con múltiples casos
        recurrence = RecurrenceEquation(
            function_name=self.procedure_name,
            # Si el primer parámetro es un array, usamos `n` como parámetro genérico
            parameter=("n" if self.array_param_name else (self.parameter_name or "n")),
            base_case_condition=base_condition,
            base_case_cost=base_cost,
            recursive_calls=self.recursive_calls,
            non_recursive_cost=("n" if self.has_merge else "O(1)")
        )
        
        return RecursionAnalysisResult(
            procedure_name=self.procedure_name,
            is_recursive=True,
            recurrence_equation=recurrence,
            base_cases=[bc[0] for bc in self.base_cases],
            recursive_cases=[f"{call.function_name}({call.depth_reduction})" 
                           for call in self.recursive_calls],
            calls_count=len(self.recursive_calls)
        )


# ============================================================================
# API PÚBLICA MEJORADA
# ============================================================================

def analyze_recursion(procedure: ProcedureNode) -> RecursionAnalysisResult:
    """
    Analiza un procedimiento para detectar recursión.
    
    Returns:
        RecursionAnalysisResult con ecuaciones para peor, mejor y caso promedio
    """
    visitor = RecursionAnalyzerVisitor(procedure.name)
    return visitor.visit_procedure(procedure)


def to_recurrence(code: str, procedure_name: Optional[str] = None, 
                  case: str = "worst") -> Optional[str]:
    """
    Extrae ecuación de recurrencia para un caso específico.
    
    Args:
        code: Código pseudocódigo
        procedure_name: Nombre del procedimiento (usa el primero si es None)
        case: "worst", "best" o "average"
        
    Returns:
        String con la ecuación o None si no es recursivo
        
    Example:
        >>> code = '''
        ... Factorial(n)
        ... begin
        ...     if (n ≤ 1) then return 1
        ...     else return n * call Factorial(n-1)
        ... end
        ... '''
        >>> print(to_recurrence(code, case="worst"))
        T(n) = T(n-1) + O(1)
        >>> print(to_recurrence(code, case="best"))
        T(n) = O(1)
    """
    from parser.parser import parse
    
    try:
        ast = parse(code)
        
        if not ast.procedures:
            return None
        
        # Usar el procedimiento especificado o el primero
        if procedure_name:
            proc = next((p for p in ast.procedures if p.name == procedure_name), None)
            if not proc:
                return None
        else:
            proc = ast.procedures[0]
        
        result = analyze_recursion(proc)
        
        if not result.is_recursive:
            return None
        
        # Retornar según el caso solicitado
        if case == "worst":
            return result.to_recurrence_worst()
        elif case == "best":
            return result.to_recurrence_best()
        elif case == "average":
            return result.to_recurrence_average()
        else:
            return result.to_recurrence_worst()
    
    except Exception:
        return None


def to_all_recurrences(code: str, procedure_name: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Extrae TODAS las ecuaciones de recurrencia.
    
    Returns:
        Dict con keys: 'worst', 'best', 'average'
        o None si no es recursivo
        
    Example:
        >>> equations = to_all_recurrences(factorial_code)
        >>> print(equations['worst'])
        T(n) = T(n-1) + O(1)
        >>> print(equations['best'])
        T(n) = O(1)
    """
    from parser.parser import parse
    
    try:
        ast = parse(code)
        
        if not ast.procedures:
            return None
        
        if procedure_name:
            proc = next((p for p in ast.procedures if p.name == procedure_name), None)
            if not proc:
                return None
        else:
            proc = ast.procedures[0]
        
        result = analyze_recursion(proc)
        return result.get_all_equations()
    
    except Exception:
        return None


def analyze_all_procedures(ast: ProgramNode) -> Dict[str, RecursionAnalysisResult]:
    """Analiza todos los procedimientos"""
    results = {}
    for procedure in ast.procedures:
        result = analyze_recursion(procedure)
        results[procedure.name] = result
    return results


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demuestra el análisis de recursión con múltiples casos"""
    from parser.parser import parse
    
    examples = {
        "Factorial (con early exit)": """
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
        """,
        
        "Fibonacci": """
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
        """,
        
        "Binary Search (simple)": """
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
    
    if (A[mid] < x) then
    begin
        return call BinarySearch(A, mid+1, right, x)
    end
    
    return call BinarySearch(A, left, mid-1, x)
end
        """,
        
        "Binary Search (con else anidado)": """
BinarySearch(A[], left, right, x)
begin
    if (left > right) then
    begin
        return -1
    end
    else
    begin
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
end
        """
    }
    
    print("="*70)
    print("ANÁLISIS DE RECURSIÓN - MÚLTIPLES CASOS")
    print("="*70)
    
    for name, code in examples.items():
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print(f"{'='*70}")
        
        try:
            equations = to_all_recurrences(code)
            
            if equations:
                print(f"\n🔴 Peor Caso:")
                print(f"   {equations['worst']}")
                
                print(f"\n🟢 Mejor Caso:")
                print(f"   {equations['best']}")
                
                print(f"\n🟡 Caso Promedio:")
                print(f"   {equations['average']}")
            else:
                print("\n❌ No es recursivo")
                
                # DEBUG: Ver qué está pasando
                from parser.parser import parse as parse_code
                ast = parse_code(code)
                proc = ast.procedures[0]
                visitor = RecursionAnalyzerVisitor(proc.name)
                result = visitor.visit_procedure(proc)
                
                print(f"\n🔍 DEBUG:")
                print(f"   - Llamadas recursivas encontradas: {len(visitor.recursive_calls)}")
                if visitor.recursive_calls:
                    for call in visitor.recursive_calls:
                        print(f"     • {call.function_name}({call.depth_reduction})")
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()