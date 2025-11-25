"""
Cliente LLM para Groq API
=========================
 echo 'export GROQ_API_KEY="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"' >> ~/.bashrc
Maneja la comunicación con Groq y parsing de respuestas.
"""

import json
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
import requests


@dataclass
class LLMResponse:
    """Respuesta estructurada del LLM"""
    
    raw_text: str
    parsed_json: Optional[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0
    
    def get(self, key: str, default=None):
        """Acceso seguro a valores del JSON"""
        if self.parsed_json:
            return self.parsed_json.get(key, default)
        return default


class GroqClient:
    """Cliente para Groq API con Llama 3.3 70B"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 2000,
        timeout: int = 30
    ) -> LLMResponse:
        """
        Envía un prompt al LLM y retorna respuesta parseada.
        
        Args:
            prompt: Prompt con el análisis a realizar
            temperature: Temperatura para generación (0.0-1.0)
            max_tokens: Máximo de tokens en respuesta
            timeout: Timeout en segundos
            
        Returns:
            LLMResponse con resultado parseado
        """
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un experto en análisis de complejidad computacional. "
                               "Respondes SIEMPRE en formato JSON válido, sin texto adicional."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return LLMResponse(
                    raw_text="",
                    parsed_json=None,
                    success=False,
                    error=f"API Error {response.status_code}: {response.text}",
                    latency_ms=latency_ms
                )
            
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"]
            
            # Intentar parsear JSON
            parsed_json = self._parse_json_response(raw_text)
            
            return LLMResponse(
                raw_text=raw_text,
                parsed_json=parsed_json,
                success=True,
                latency_ms=latency_ms
            )
        
        except requests.Timeout:
            return LLMResponse(
                raw_text="",
                parsed_json=None,
                success=False,
                error=f"Timeout después de {timeout}s",
                latency_ms=(time.time() - start_time) * 1000
            )
        
        except Exception as e:
            return LLMResponse(
                raw_text="",
                parsed_json=None,
                success=False,
                error=f"Error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parsea respuesta JSON del LLM (maneja markdown y texto extra).
        
        Args:
            text: Texto de respuesta del LLM
            
        Returns:
            Dict parseado o None si falla
        """
        # Limpiar markdown
        text = text.strip()
        
        # Remover bloques de código markdown
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end]
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end]
        
        # Buscar JSON entre llaves
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        
        # Intentar parsear
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"⚠ JSON parse error: {e}")
            print(f"  Text: {text[:200]}...")
            return None


# ============================================================================
# TEST DEL CLIENTE
# ============================================================================

def test_client():
    """Test básico del cliente Groq"""
    import os
    
    print("="*70)
    print("TEST DEL CLIENTE GROQ")
    print("="*70)
    
    # Obtener API key
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("\n❌ GROQ_API_KEY no configurada")
        print("\n💡 Configura tu API key:")
        print("   export GROQ_API_KEY='tu-api-key'")
        return False
    
    # Crear cliente
    client = GroqClient(api_key)
    
    # Prompt de prueba simple
    test_prompt = """Analiza este algoritmo y responde en JSON:

```
BubbleSort(A[], n)
begin
    for i ← 1 to n-1 do
    begin
        for j ← 1 to n-i do
        begin
            if (A[j] > A[j+1]) then
            begin
                swap(A[j], A[j+1])
            end
        end
    end
end
```

Formato de respuesta:
{
  "worst_case": "O(...)",
  "best_case": "Ω(...)",
  "explanation": "..."
}"""
    
    print("\n📤 Enviando prompt de prueba...")
    print(f"   Modelo: {client.model}")
    
    response = client.analyze(test_prompt, temperature=0.1)
    
    print(f"\n⏱️  Latencia: {response.latency_ms:.0f}ms")
    
    if response.success:
        print("\n✓ Respuesta recibida exitosamente")
        
        if response.parsed_json:
            print("\n📊 JSON Parseado:")
            print(json.dumps(response.parsed_json, indent=2))
            
            # Validar campos esperados
            if "worst_case" in response.parsed_json:
                print("\n✓ Campo 'worst_case' presente")
            if "best_case" in response.parsed_json:
                print("\n✓ Campo 'best_case' presente")
            
            return True
        else:
            print("\n⚠ Respuesta no se pudo parsear como JSON")
            print(f"\nRespuesta raw:\n{response.raw_text[:500]}...")
            return False
    else:
        print(f"\n❌ Error: {response.error}")
        return False


if __name__ == "__main__":
    import sys
    success = test_client()
    sys.exit(0 if success else 1)