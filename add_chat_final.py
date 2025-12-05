"""
Agregar funcionalidad de chat a app_complete.py de forma segura
"""

with open('app_complete.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Agregar imports del ChatAnalyzer después de los imports de LLM
for i, line in enumerate(lines):
    if 'LLM_AVAILABLE = False' in line:
        # Insertar después de esta línea
        lines.insert(i + 1, '\n')
        lines.insert(i + 2, '# Importar chat analyzer\n')
        lines.insert(i + 3, 'try:\n')
        lines.insert(i + 4, '    from src.llm.llm_chat_analyzer import ChatAnalyzer\n')
        lines.insert(i + 5, '    from src.visualization.chat_display import display_llm_chat_analysis\n')
        lines.insert(i + 6, '    CHAT_ANALYZER_AVAILABLE = True\n')
        lines.insert(i + 7, 'except ImportError:\n')
        lines.insert(i + 8, '    CHAT_ANALYZER_AVAILABLE = False\n')
        break

# 2. Agregar toggle del chat en el sidebar (después de la sección de sintaxis)
for i, line in enumerate(lines):
    if '# MAIN: ENTRADA DE CÓDIGO' in line:
        # Insertar antes de esta línea
        lines.insert(i, '    # ========================================================================\n')
        lines.insert(i + 1, '    # CHAT CONVERSACIONAL\n')
        lines.insert(i + 2, '    # ========================================================================\n')
        lines.insert(i + 3, '    \n')
        lines.insert(i + 4, '    if LLM_AVAILABLE and CHAT_ANALYZER_AVAILABLE:\n')
        lines.insert(i + 5, '        st.divider()\n')
        lines.insert(i + 6, '        st.header("💬 Chat con IA")\n')
        lines.insert(i + 7, '        \n')
        lines.insert(i + 8, '        enable_chat = st.toggle(\n')
        lines.insert(i + 9, '            "Mostrar Análisis Conversacional",\n')
        lines.insert(i + 10, '            value=True,\n')
        lines.insert(i + 11, '            help="El asistente de IA explicará tu algoritmo de forma conversacional"\n')
        lines.insert(i + 12, '        )\n')
        lines.insert(i + 13, '        \n')
        lines.insert(i + 14, '        st.session_state[\'chat_enabled\'] = enable_chat\n')
        lines.insert(i + 15, '        \n')
        lines.insert(i + 16, '        if enable_chat:\n')
        lines.insert(i + 17, '            st.info("💡 **Chat habilitado**: Recibirás explicaciones conversacionales")\n')
        lines.insert(i + 18, '        else:\n')
        lines.insert(i + 19, '            st.info("ℹ️ Chat deshabilitado")\n')
        lines.insert(i + 20, '    else:\n')
        lines.insert(i + 21, '        st.session_state[\'chat_enabled\'] = False\n')
        lines.insert(i + 22, '\n')
        lines.insert(i + 23, '\n')
        break

# 3. Agregar sección de chat después de display_procedure_analysis (en la sección CON LLM)
for i, line in enumerate(lines):
    if '# Mostrar resultados con validación LLM' in line:
        # Buscar la línea de display_procedure_analysis después de este comentario
        for j in range(i, min(i + 10, len(lines))):
            if 'display_procedure_analysis(results)' in lines[j]:
                # Insertar después de esta línea
                lines.insert(j + 1, '                        \n')
                lines.insert(j + 2, '                        # CHAT CONVERSACIONAL\n')
                lines.insert(j + 3, '                        chat_enabled = st.session_state.get(\'chat_enabled\', False)\n')
                lines.insert(j + 4, '                        if chat_enabled and CHAT_ANALYZER_AVAILABLE:\n')
                lines.insert(j + 5, '                            try:\n')
                lines.insert(j + 6, '                                st.divider()\n')
                lines.insert(j + 7, '                                with st.spinner("🤖 Generando análisis conversacional..."):\n')
                lines.insert(j + 8, '                                    chat_analyzer = ChatAnalyzer(api_key)\n')
                lines.insert(j + 9, '                                    first_proc = next(iter(results.values()))\n')
                lines.insert(j + 10, '                                    static_complexity = {\n')
                lines.insert(j + 11, '                                        \'worst_case\': getattr(first_proc, \'final_worst\', \'O(?)\'),\n')
                lines.insert(j + 12, '                                        \'algorithm_type\': getattr(first_proc, \'algorithm_type\', \'unknown\')\n')
                lines.insert(j + 13, '                                    }\n')
                lines.insert(j + 14, '                                    chat_messages = chat_analyzer.analyze(code_input, static_complexity)\n')
                lines.insert(j + 15, '                                    display_llm_chat_analysis(chat_messages)\n')
                lines.insert(j + 16, '                            except Exception as e:\n')
                lines.insert(j + 17, '                                st.warning(f"⚠️ No se pudo generar análisis conversacional: {e}")\n')
                break
        break

# Guardar
with open('app_complete.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ app_complete.py actualizado con funcionalidad de chat")
print("   - Imports agregados")
print("   - Toggle en sidebar agregado")
print("   - Sección de chat agregada")
