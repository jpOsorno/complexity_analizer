"""API pequeña para obtener ecuaciones de recurrencia.

Funciones exportadas:
- `get_recurrences(code, procedure_name=None)` -> Dict con keys 'worst','best','average'
- `get_recurrence(code, procedure_name=None, case='worst')` -> ecuación única

También soporta ejecución como script para pruebas rápidas.
"""

from typing import Optional, Dict

from recursion_analyzer import to_recurrence, to_all_recurrences
from recurrence_solver import solve_recurrence


def get_recurrences(code: str, procedure_name: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Retorna las tres ecuaciones de recurrencia (worst,best,average).

    Args:
        code: Código en pseudocódigo (string)
        procedure_name: Nombre del procedimiento a analizar (opcional)

    Returns:
        Dict con keys 'worst','best','average' o None si no es recursivo o hay error.
    """
    return to_all_recurrences(code, procedure_name=procedure_name)


def get_recurrence(code: str, procedure_name: Optional[str] = None, case: str = 'worst') -> Optional[str]:
    """Retorna la ecuación de recurrencia para un caso específico.

    Args:
        code: Código en pseudocódigo
        procedure_name: Nombre del procedimiento (opcional)
        case: 'worst' | 'best' | 'average'
    """
    return to_recurrence(code, procedure_name=procedure_name, case=case)


if __name__ == '__main__':
    # Demo rápido (misma muestra que antes)
    sample = """
BinarySearch(A[], left, right, x)
begin
    if (left > right) then
    begin
        return -1
    end
    else
    begin
        mid ← floor((left + right) / 2)
        return call BinarySearch(A, mid+1, right, x)
    end
end

    """

    print('Ejemplo:')
    eqs = get_recurrences(sample)
    if eqs:
        # Para cada caso mostramos la ecuación y cómo se resolvió
        for label, title in (('worst', 'Worst'), ('best', 'Best'), ('average', 'Avg')):
            eq = eqs.get(label)
            if not eq:
                print(f'  {title:6}: <no encontrado>')
                continue
            print(f'  {title:6}: {eq}')

            try:
                solution = solve_recurrence(eq)
                # Imprimir resultado con sangría para facilitar lectura
                sol_lines = str(solution).splitlines()
                print('    Solución y pasos:')
                for line in sol_lines:
                    print('      ' + line)
            except Exception as e:
                print('    No se pudo resolver automáticamente:', e)
    else:
        print('No es recursivo o hubo un error en el análisis.')