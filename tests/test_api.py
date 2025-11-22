# Guarda cada algoritmo en un archivo separado o todos juntos
# Luego usa tu parser para analizarlos

from src.parser.parser import parse
from src.analyzer.recursion_analyzer import to_all_recurrences
from src.analyzer.recurrence_solver import solve_recurrence

# Para cada algoritmo:
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

# Obtener ecuaciones
equations = to_all_recurrences(code)
print("Ecuaciones detectadas:")
print(f"  Peor: {equations['worst']}")
print(f"  Mejor: {equations['best']}")
print(f"  Promedio: {equations['average']}")

# Resolver cada ecuación
for case, eq in equations.items():
    if eq:
        solution = solve_recurrence(eq)
        print(f"\n{case.upper()}:")
        print(solution)