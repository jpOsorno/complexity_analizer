# Script para agregar display de resultados LLM
import sys

# Leer el archivo
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Encontrar la línea donde insertar
insert_index = None
for i, line in enumerate(lines):
    if 'st.divider()' in line and i > 340 and i < 360:
        # Verificar que la siguiente línea esté vacía o sea el except
        if i + 1 < len(lines) and ('except' in lines[i+2] or lines[i+1].strip() == ''):
            insert_index = i + 1
            break

if insert_index:
    # Insertar las nuevas líneas
    new_lines = [
        '                        st.header("📊 Resultados del Análisis con IA")\n',
        '                        \n',
        '                        # Mostrar resultados con validación LLM\n',
        '                        display_procedure_analysis(results)\n',
        '                        \n'
    ]
    
    lines[insert_index:insert_index] = new_lines
    
    # Escribir el archivo
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✓ Líneas insertadas en posición {insert_index}")
else:
    print("✗ No se encontró la posición correcta")
    sys.exit(1)
