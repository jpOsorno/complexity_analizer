from src.llm.unified_analyzer_llm import analyze_with_llm

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

# Análisis con validación LLM
results = analyze_with_llm(code, enable_llm=True)

for name, result in results.items():
    print(result)  # Muestra análisis + validación LLM