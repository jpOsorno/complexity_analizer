from recursion_analyzer import to_recurrence

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

# Obtener solo la ecuación
equation = to_recurrence(code)
print(equation)
# → "T(n) = T(n-1) + O(1)"