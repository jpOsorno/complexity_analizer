# Patch para agregar display de resultados LLM
with open('app.py', 'rb') as f:
    content = f.read().decode('utf-8-sig')

# Buscar la línea donde insertar
search_str = 'st.success("✅ Análisis completado (con validación IA)")\r\n                        \r\n                        st.divider()'
replacement_str = '''st.success("✅ Análisis completado (con validación IA)")
                        
                        st.divider()
                        st.header("📊 Resultados del Análisis con IA")
                        
                        # Mostrar resultados con validación LLM
                        display_procedure_analysis(results)'''

if search_str in content:
    content = content.replace(search_str, replacement_str)
    with open('app.py', 'wb') as f:
        f.write(content.encode('utf-8-sig'))
    print("✓ Archivo actualizado correctamente")
else:
    print("✗ No se encontró el patrón de búsqueda")
    print("Buscando alternativa...")
    # Intentar sin \r
    search_str2 = 'st.success("✅ Análisis completado (con validación IA)")\n                        \n                        st.divider()'
    if search_str2 in content:
        content = content.replace(search_str2, replacement_str)
        with open('app.py', 'wb') as f:
            f.write(content.encode('utf-8-sig'))
        print("✓ Archivo actualizado correctamente (alternativa)")
    else:
        print("✗ Tampoco se encontró la alternativa")
