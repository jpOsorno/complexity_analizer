# Script para agregar display de resultados LLM en app.py
import re

# Leer archivo
with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Buscar el patrón y reemplazar
pattern = r'(st\.success\("✅ Análisis completado \(con validación IA\)"\)\s+st\.divider\(\))'
replacement = r'''st.success("✅ Análisis completado (con validación IA)")
                        
                        st.divider()
                        st.header("📊 Resultados del Análisis con IA")
                        
                        # Mostrar resultados con validación LLM
                        display_procedure_analysis(results)
                        
                        st.divider()'''

# Hacer el reemplazo
new_content = re.sub(pattern, replacement, content, count=1)

# Verificar que se hizo el cambio
if new_content != content:
    # Guardar
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✓ Archivo actualizado correctamente")
    print("✓ Se agregó display_procedure_analysis(results)")
else:
    print("✗ No se encontró el patrón para reemplazar")
    print("Intentando búsqueda manual...")
    
    # Buscar manualmente
    if "Análisis completado (con validación IA)" in content:
        print("✓ Encontrado el texto de éxito")
        # Encontrar la posición
        idx = content.find('st.success("✅ Análisis completado (con validación IA)")')
        if idx != -1:
            # Encontrar el siguiente st.divider()
            divider_idx = content.find('st.divider()', idx)
            if divider_idx != -1:
                # Insertar después del divider
                insert_pos = divider_idx + len('st.divider()')
                new_content = (content[:insert_pos] + 
                             '\n                        st.header("📊 Resultados del Análisis con IA")\n' +
                             '                        display_procedure_analysis(results)\n' +
                             '                        \n' +
                             content[insert_pos:])
                
                with open('app.py', 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print("✓ Archivo actualizado con método alternativo")
    else:
        print("✗ No se encontró el texto esperado")
