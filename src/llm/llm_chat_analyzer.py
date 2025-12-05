"""
Analizador Conversacional con LLM
==================================

Proporciona análisis de algoritmos en formato de chat conversacional.
"""

import sys
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.llm.llm_client import GroqClient, LLMResponse


@dataclass
class ChatMessage:
    """Mensaje del chat de análisis"""
    text: str
    message_type: str  # 'identification', 'pattern', 'complexity', 'optimization', 'insight'
    confidence: float = 1.0
    code_snippet: Optional[str] = None
    emoji: str = "💬"


class ChatAnalyzer:
    """
    Analizador conversacional que usa LLM para explicar algoritmos
    de forma natural y amigable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el analizador conversacional.
        
        Args:
            api_key: API key de Groq (si None, busca en env)
        """
        if api_key is None:
            api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            raise ValueError("API key no proporcionada. Configura GROQ_API_KEY.")
        
        self.client = GroqClient(api_key)
        
        # Emojis por tipo de mensaje
        self.emojis = {
            'identification': '🔍',
            'pattern': '🧩',
            'complexity': '📊',
            'optimization': '💡',
            'insight': '✨',
            'warning': '⚠️',
            'strength': '💪'
        }
    
    def analyze(
        self, 
        code: str, 
        static_complexity: Optional[Dict] = None
    ) -> List[ChatMessage]:
        """
        Analiza el código y genera mensajes conversacionales.
        
        Args:
            code: Código pseudocódigo
            static_complexity: Resultados del análisis estático (opcional)
            
        Returns:
            Lista de ChatMessage con el análisis conversacional
        """
        # Construir prompt conversacional
        prompt = self._build_conversational_prompt(code, static_complexity)
        
        # Llamar al LLM
        response = self.client.analyze(
            prompt,
            temperature=0.7,  # Más creativo para conversación
            max_tokens=2000
        )
        
        if not response.success:
            return [ChatMessage(
                text=f"❌ Error al generar análisis: {response.error}",
                message_type='error',
                confidence=0.0,
                emoji='❌'
            )]
        
        # Parsear respuesta en mensajes
        messages = self._parse_response_to_messages(response)
        
        return messages
    
    def _build_conversational_prompt(
        self, 
        code: str, 
        static_complexity: Optional[Dict]
    ) -> str:
        """Construye prompt conversacional para el LLM"""
        
        prompt = f"""Eres un asistente experto en algoritmos que explica de forma conversacional y amigable.

Analiza este algoritmo y proporciona un análisis conversacional en español:

```
{code}
```
"""
        
        if static_complexity:
            prompt += f"""
Análisis estático disponible:
- Complejidad detectada: {static_complexity.get('worst_case', 'N/A')}
- Tipo: {static_complexity.get('algorithm_type', 'N/A')}
"""
        
        prompt += """
Genera tu respuesta en formato JSON con esta estructura:

{
  "identification": "Identificación del algoritmo en 1-2 oraciones conversacionales (ej: '¡Veo que estás analizando un algoritmo de ordenamiento! Específicamente, esto es un Bubble Sort...')",
  "pattern": "Explicación del patrón algorítmico en lenguaje natural (ej: 'Este algoritmo usa un enfoque de comparación e intercambio. Básicamente, compara elementos adyacentes...')",
  "complexity_explanation": "Explicación conversacional de la complejidad (ej: 'En cuanto a eficiencia, este algoritmo tiene complejidad O(n²). ¿Qué significa esto? Bueno, si duplicas el tamaño de tu array...')",
  "strengths": "Fortalezas del algoritmo (ej: 'Una ventaja de este algoritmo es que es muy simple de entender e implementar...')",
  "optimizations": "Sugerencias de optimización en tono amigable (ej: 'Si quieres mejorar el rendimiento, podrías considerar...')",
  "insights": "Insights adicionales o curiosidades (ej: 'Dato curioso: este algoritmo se llama Bubble Sort porque los elementos más grandes \"burbujean\" hacia arriba...')"
}

Sé conversacional, amigable y educativo. Usa analogías cuando sea apropiado.
"""
        
        return prompt
    
    def _parse_response_to_messages(self, response: LLMResponse) -> List[ChatMessage]:
        """Parsea la respuesta del LLM en mensajes de chat"""
        
        messages = []
        
        if not response.parsed_json:
            # Fallback: usar texto raw
            return [ChatMessage(
                text=response.raw_text,
                message_type='general',
                emoji='💬'
            )]
        
        data = response.parsed_json
        
        # Mensaje de identificación
        if data.get('identification'):
            messages.append(ChatMessage(
                text=data['identification'],
                message_type='identification',
                emoji=self.emojis['identification']
            ))
        
        # Mensaje de patrón
        if data.get('pattern'):
            messages.append(ChatMessage(
                text=data['pattern'],
                message_type='pattern',
                emoji=self.emojis['pattern']
            ))
        
        # Mensaje de complejidad
        if data.get('complexity_explanation'):
            messages.append(ChatMessage(
                text=data['complexity_explanation'],
                message_type='complexity',
                emoji=self.emojis['complexity']
            ))
        
        # Fortalezas
        if data.get('strengths'):
            messages.append(ChatMessage(
                text=data['strengths'],
                message_type='strength',
                emoji=self.emojis['strength']
            ))
        
        # Optimizaciones
        if data.get('optimizations'):
            messages.append(ChatMessage(
                text=data['optimizations'],
                message_type='optimization',
                emoji=self.emojis['optimization']
            ))
        
        # Insights
        if data.get('insights'):
            messages.append(ChatMessage(
                text=data['insights'],
                message_type='insight',
                emoji=self.emojis['insight']
            ))
        
        return messages


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demo del analizador conversacional"""
    
    print("="*70)
    print("DEMO: ANALIZADOR CONVERSACIONAL CON IA")
    print("="*70)
    
    # Verificar API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("\n❌ GROQ_API_KEY no configurada")
        print("\n💡 Configura tu API key:")
        print("   export GROQ_API_KEY='tu-api-key'")
        return False
    
    # Crear analizador
    analyzer = ChatAnalyzer(api_key)
    
    # Código de prueba
    test_code = """BubbleSort(A[], n)
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
end"""
    
    print("\n📝 Código a analizar:")
    print("-" * 70)
    print(test_code)
    print("-" * 70)
    
    print("\n🤖 Generando análisis conversacional...")
    
    start_time = time.time()
    messages = analyzer.analyze(test_code)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Tiempo: {elapsed:.2f}s")
    print(f"📨 Mensajes generados: {len(messages)}")
    
    print("\n" + "="*70)
    print("CHAT DE ANÁLISIS")
    print("="*70)
    
    for i, msg in enumerate(messages, 1):
        print(f"\n{msg.emoji} Mensaje {i} ({msg.message_type}):")
        print("-" * 70)
        print(msg.text)
        if msg.code_snippet:
            print(f"\n```\n{msg.code_snippet}\n```")
    
    print("\n" + "="*70)
    print("✅ Demo completado")
    print("="*70)
    
    return True


if __name__ == "__main__":
    import sys
    success = demo()
    sys.exit(0 if success else 1)
