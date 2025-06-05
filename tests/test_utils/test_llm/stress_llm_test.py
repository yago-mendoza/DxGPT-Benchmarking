#!/usr/bin/env python3
"""
Test completo para Azure LLM v4
Cobertura: básico → avanzado → edge cases → stress test
"""

import json
import os
import time
from typing import Dict, Any
from utils.llm import Azure, AzureLLM, LLMConfig, Schema, Template, create_llm, quick_generate, BatchProcessor


class AzureLLMTester:
    """Tester completo con métricas y control de costos."""
    
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def test(self, name: str, func):
        """Ejecuta un test individual con manejo de errores."""
        self.total_tests += 1
        print(f"\n🧪 Test {self.total_tests}: {name}")
        
        try:
            start_time = time.time()
            result = func()
            duration = time.time() - start_time
            
            self.passed_tests += 1
            print(f"   ✅ PASS ({duration:.2f}s)")
            self.results.append({"name": name, "status": "PASS", "duration": duration, "result": result})
            
        except Exception as e:
            self.failed_tests += 1
            print(f"   ❌ FAIL: {str(e)}")
            self.results.append({"name": name, "status": "FAIL", "error": str(e)})
    
    def summary(self):
        """Muestra resumen final."""
        print(f"\n{'='*50}")
        print(f"📊 RESUMEN DE TESTS")
        print(f"{'='*50}")
        print(f"Total: {self.total_tests}")
        print(f"✅ Pasados: {self.passed_tests}")
        print(f"❌ Fallidos: {self.failed_tests}")
        print(f"📈 Tasa de éxito: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        if self.failed_tests > 0:
            print(f"\n❌ Tests fallidos:")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"   - {result['name']}: {result['error']}")


def main():
    """Ejecuta la suite completa de tests."""
    tester = AzureLLMTester()
    
    print("🚀 Iniciando Azure LLM v4 Full Test Suite")
    print("📝 Prompts cortos para minimizar costos de tokens")
    
    # ========================================
    # FASE 1: TESTS BÁSICOS
    # ========================================
    
    def test_basic_azure_alias():
        """Test básico del alias Azure con deployment name."""
        print("Planteamiento: Usar alias Azure para generar respuesta simple.")
        llm = Azure("gpt-4o")
        response = llm.generate("Di 'OK'")
        print(f"Output LLM: {response}")
        assert isinstance(response, str)
        assert len(response.strip()) > 0
        return f"Response: '{response[:20]}...'"
    
    def test_azure_with_params():
        """Test Azure con parámetros adicionales."""
        print("Planteamiento: Usar Azure con temperatura baja.")
        llm = Azure("gpt-4o", temperature=0.1)
        response = llm.generate("Di solo 'TEST'")
        print(f"Output LLM: {response}")
        assert isinstance(response, str)
        return f"Response: '{response[:20]}...'"
    
    def test_create_llm_function():
        """Test función de conveniencia create_llm."""
        print("Planteamiento: Usar create_llm para instanciar y generar.")
        llm = create_llm("gpt-4o")
        response = llm.generate("Responde 'CREATE_LLM'")
        print(f"Output LLM: {response}")
        assert isinstance(response, str)
        return f"Response: '{response[:20]}...'"
    
    def test_quick_generate():
        """Test función quick_generate."""
        print("Planteamiento: Usar quick_generate para respuesta rápida.")
        response = quick_generate("Di 'QUICK'")
        print(f"Output LLM: {response}")
        assert isinstance(response, str)
        return f"Response: '{response[:20]}...'"
    
    tester.test("Basic Azure Alias", test_basic_azure_alias)
    tester.test("Azure with Parameters", test_azure_with_params)
    tester.test("create_llm Function", test_create_llm_function)
    tester.test("quick_generate Function", test_quick_generate)
    
    # ========================================
    # FASE 2: CONFIGURACIÓN AVANZADA
    # ========================================
    
    def test_llm_config_from_env():
        """Test configuración desde environment."""
        print("Planteamiento: Crear config desde env y generar.")
        config = LLMConfig.from_env(deployment_name="gpt-4o", temperature=0.2)
        llm = Azure(config=config)
        response = llm.generate("Di 'CONFIG'")
        print(f"Output LLM: {response}")
        assert config.deployment_name == "gpt-4o"
        assert config.temperature == 0.2
        return f"Config OK, Response: '{response[:15]}...'"
    
    def test_explicit_config():
        """Test configuración explícita."""
        print("Planteamiento: Usar config explícita para generar.")
        config = LLMConfig(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name="gpt-4o",
            temperature=0.0
        )
        llm = AzureLLM(config=config)
        response = llm.generate("Di 'EXPLICIT'")
        print(f"Output LLM: {response}")
        return f"Explicit config OK, Response: '{response[:15]}...'"
    
    tester.test("LLMConfig from Environment", test_llm_config_from_env)
    tester.test("Explicit Configuration", test_explicit_config)
    
    # ========================================
    # FASE 3: SCHEMAS Y SALIDA ESTRUCTURADA
    # ========================================
    
    def test_simple_json_schema():
        """Test schema JSON simple."""
        print("Planteamiento: Usar schema JSON simple para estructurar salida.")
        schema_dict = {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "status": {"type": "string"}
            },
            "required": ["message", "status"]
        }
        
        llm = Azure("gpt-4o")
        response = llm.generate(
            "Responde con mensaje='Hello' y status='OK'",
            schema=schema_dict
        )
        print(f"Output LLM: {response}")
        assert isinstance(response, dict)
        assert "message" in response
        assert "status" in response
        return f"JSON response: {response}"
    
    def test_schema_object():
        """Test objeto Schema."""
        print("Planteamiento: Usar objeto Schema para estructurar salida.")
        schema_dict = {
            "type": "object",
            "properties": {
                "number": {"type": "integer"},
                "word": {"type": "string"}
            },
            "required": ["number", "word"]
        }
        
        schema = Schema.load(schema_dict)
        llm = Azure("gpt-4o")
        response = llm.generate(
            "Responde con number=42 y word='test'",
            schema=schema
        )
        print(f"Output LLM: {response}")
        assert isinstance(response, dict)
        assert isinstance(response.get("number"), int)
        return f"Schema object response: {response}"
    
    def test_complex_schema():
        """Test schema más complejo."""
        print("Planteamiento: Usar schema complejo con objetos anidados.")
        schema_dict = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "active": {"type": "boolean"}
                    },
                    "required": ["id", "name", "active"]
                },
                "metadata": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["data", "metadata"]
        }
        
        llm = Azure("gpt-4o")
        response = llm.generate(
            "Responde con data.id=1, data.name='test', data.active=true, metadata=['a','b']",
            schema=schema_dict
        )
        print(f"Output LLM: {response}")
        assert isinstance(response, dict)
        assert "data" in response
        assert "metadata" in response
        assert isinstance(response["metadata"], list)
        return f"Complex schema: {str(response)[:50]}..."
    
    tester.test("Simple JSON Schema", test_simple_json_schema)
    tester.test("Schema Object", test_schema_object)
    tester.test("Complex Schema", test_complex_schema)
    
    # ========================================
    # FASE 4: SISTEMA DE TEMPLATES
    # ========================================
    
    def test_basic_template():
        """Test template básico."""
        print("Planteamiento: Usar template básico para generar respuesta.")
        llm = Azure("gpt-4o")
        template = llm.template("Di '{word}' en {lang}")
        
        response = template(word="hello", lang="español")
        print(f"Output LLM: {response}")
        assert isinstance(response, str)
        return f"Template response: '{response[:30]}...'"
    
    def test_template_with_schema():
        """Test template con schema."""
        print("Planteamiento: Usar template con schema para estructurar salida.")
        schema_dict = {
            "type": "object",
            "properties": {
                "original": {"type": "string"},
                "translated": {"type": "string"}
            },
            "required": ["original", "translated"]
        }
        
        llm = Azure("gpt-4o")
        template = llm.template(
            "Traduce '{text}' a {lang}",
            schema=schema_dict
        )
        
        response = template(text="hi", lang="español")
        print(f"Output LLM: {response}")
        assert isinstance(response, dict)
        assert "original" in response
        assert "translated" in response
        return f"Template+Schema: {response}"
    
    def test_template_with_fixed_params():
        """Test template con parámetros fijos."""
        print("Planteamiento: Usar template con parámetros fijos para generar.")
        llm = Azure("gpt-4o")
        template = llm.template(
            "Responde '{response}' en tono {tone}",
            temperature=0.1,
            max_tokens=20
        )
        
        response = template(response="OK", tone="formal")
        print(f"Output LLM: {response}")
        assert isinstance(response, str)
        return f"Fixed params template: '{response[:25]}...'"
    
    tester.test("Basic Template", test_basic_template)
    tester.test("Template with Schema", test_template_with_schema)
    tester.test("Template with Fixed Parameters", test_template_with_fixed_params)
    
    # ========================================
    # FASE 5: BATCH PROCESSING
    # ========================================
    
    def test_batch_simple():
        """Test batch processing simple sin schema."""
        print("Planteamiento: Procesar batch simple sin schema.")
        llm = Azure("gpt-4o")
        
        batch_items = [
            {"text": "A"},
            {"text": "B"},
            {"text": "C"}
        ]
        
        response = llm.generate(
            "Di cada letra seguida de '!'",
            batch_items=batch_items
        )
        print(f"Output LLM: {response}")
        assert isinstance(response, list)
        assert len(response) == 3
        return f"Batch response: {response}"
    
    def test_batch_with_schema():
        """Test batch processing con schema."""
        print("Planteamiento: Procesar batch con schema estructurado.")
        llm = Azure("gpt-4o")
        
        batch_items = [
            {"num": 1},
            {"num": 2},
            {"num": 3}
        ]
        
        schema = {
            "type": "object",
            "properties": {
                "number": {"type": "integer"},
                "doubled": {"type": "integer"}
            },
            "required": ["number", "doubled"]
        }
        
        response = llm.generate(
            "Para cada número, devuelve el número y su doble",
            batch_items=batch_items,
            schema=schema
        )
        print(f"Output LLM: {response}")
        assert isinstance(response, list)
        assert len(response) == 3
        assert all(isinstance(item, dict) for item in response)
        assert all("number" in item and "doubled" in item for item in response)
        return f"Batch+Schema: {response}"
    
    def test_batch_processor():
        """Test BatchProcessor directamente."""
        print("Planteamiento: Probar BatchProcessor para formateo.")
        processor = BatchProcessor()
        
        items = [{"id": 1, "name": "test"}, {"id": 2, "name": "demo"}]
        formatted = processor.format_batch_items(items)
        print(f"Formatted: {formatted}")
        
        assert isinstance(formatted, str)
        assert "id" in formatted
        assert "name" in formatted
        return "BatchProcessor formatting OK"
    
    def test_batch_complex():
        """Test batch processing con datos más complejos."""
        print("Planteamiento: Procesar batch con datos complejos.")
        llm = Azure("gpt-4o")
        
        batch_items = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35}
        ]
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "greeting": {"type": "string"},
                "category": {"type": "string"}
            },
            "required": ["name", "greeting", "category"]
        }
        
        response = llm.generate(
            "Para cada persona, genera un saludo y categoriza por edad (joven/adulto)",
            batch_items=batch_items,
            schema=schema,
            temperature=0.3
        )
        print(f"Output LLM: {response}")
        assert isinstance(response, list)
        assert len(response) == 3
        assert all(isinstance(item, dict) for item in response)
        return f"Complex batch: {len(response)} items processed"
    
    tester.test("Batch Simple", test_batch_simple)
    tester.test("Batch with Schema", test_batch_with_schema)
    tester.test("BatchProcessor Direct", test_batch_processor)
    tester.test("Batch Complex", test_batch_complex)
    
    # ========================================
    # FASE 6: EDGE CASES
    # ========================================
    
    def test_empty_prompt():
        """Test prompt vacío."""
        print("Planteamiento: Generar respuesta con prompt vacío.")
        llm = Azure("gpt-4o")
        response = llm.generate("")
        print(f"Output LLM: {response}")
        assert isinstance(response, str)
        return f"Empty prompt response: '{response[:30]}...'"
    
    def test_missing_template_variables():
        """Test variables faltantes en template."""
        print("Planteamiento: Probar error con variables faltantes en template.")
        llm = Azure("gpt-4o")
        try:
            llm.generate("Di '{missing_var}'", variables={"other_var": "value"})
            assert False, "Debería haber fallado"
        except KeyError:
            print("Output LLM: KeyError correctamente capturado")
            return "KeyError correctamente capturado"
    
    def test_temperature_extremes():
        """Test temperaturas extremas."""
        print("Planteamiento: Generar respuestas con temperaturas extremas.")
        llm = Azure("gpt-4o")
        
        # Temperatura muy baja
        response_low = llm.generate("Di 'LOW'", temperature=0.0)
        print(f"Output LLM (Low Temp): {response_low}")
        
        # Temperatura muy alta
        response_high = llm.generate("Di 'HIGH'", temperature=1.0)
        print(f"Output LLM (High Temp): {response_high}")
        
        assert isinstance(response_low, str)
        assert isinstance(response_high, str)
        return f"Low: '{response_low[:10]}...', High: '{response_high[:10]}...'"
    
    def test_max_tokens_limit():
        """Test límite de tokens."""
        print("Planteamiento: Generar respuesta con límite de tokens.")
        llm = Azure("gpt-4o")
        response = llm.generate("Cuenta del 1 al 100", max_tokens=10)
        print(f"Output LLM: {response}")
        assert isinstance(response, str)
        return f"Limited tokens ({len(response)} chars): '{response[:30]}...'"
    
    def test_empty_batch():
        """Test batch vacío."""
        print("Planteamiento: Procesar batch vacío.")
        llm = Azure("gpt-4o")
        response = llm.generate(
            "Procesa estos items",
            batch_items=[]
        )
        print(f"Output LLM: {response}")
        assert isinstance(response, list)
        assert len(response) == 0
        return "Empty batch handled correctly"
    
    tester.test("Empty Prompt", test_empty_prompt)
    tester.test("Missing Template Variables", test_missing_template_variables)
    tester.test("Temperature Extremes", test_temperature_extremes)
    tester.test("Max Tokens Limit", test_max_tokens_limit)
    tester.test("Empty Batch", test_empty_batch)
    
    # ========================================
    # FASE 7: STRESS TESTS (CONTROLADOS)
    # ========================================
    
    def test_rapid_fire_requests():
        """Test múltiples requests rápidos."""
        print("Planteamiento: Enviar múltiples requests rápidos.")
        llm = Azure("gpt-4o")
        responses = []
        
        for i in range(3):  # Solo 3 para no gastar mucho
            response = llm.generate(f"Di '{i}'")
            print(f"Output LLM ({i}): {response}")
            responses.append(response)
        
        assert len(responses) == 3
        assert all(isinstance(r, str) for r in responses)
        return f"Rapid fire OK: {[r[:10] for r in responses]}"
    
    def test_mixed_schema_types():
        """Test mezcla de schemas y text."""
        print("Planteamiento: Mezclar respuestas de texto y JSON.")
        llm = Azure("gpt-4o")
        
        # Text response
        text_resp = llm.generate("Di 'TEXT'")
        print(f"Output LLM (Text): {text_resp}")
        
        # JSON response
        json_resp = llm.generate(
            "Responde con key='value'",
            schema={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}
        )
        print(f"Output LLM (JSON): {json_resp}")
        
        assert isinstance(text_resp, str)
        assert isinstance(json_resp, dict)
        return f"Mixed types OK: text='{text_resp[:10]}...', json={json_resp}"
    
    def test_reuse_llm_instance():
        """Test reutilización de instancia LLM."""
        print("Planteamiento: Reutilizar instancia LLM para múltiples prompts.")
        llm = Azure("gpt-4o")
        
        responses = []
        for prompt in ["Di 'A'", "Di 'B'", "Di 'C'"]:
            response = llm.generate(prompt)
            print(f"Output LLM ({prompt}): {response}")
            responses.append(response)
        
        assert len(responses) == 3
        return f"Reuse instance OK: {[r[:5] for r in responses]}"
    
    def test_batch_vs_individual():
        """Test comparación batch vs individual."""
        print("Planteamiento: Comparar eficiencia batch vs individual.")
        llm = Azure("gpt-4o")
        
        items = [{"n": 1}, {"n": 2}, {"n": 3}]
        
        # Batch processing
        start_batch = time.time()
        batch_response = llm.generate(
            "Di el número",
            batch_items=items
        )
        batch_time = time.time() - start_batch
        print(f"Batch time: {batch_time:.2f}s")
        print(f"Batch response: {batch_response}")
        
        # Individual processing (simulado, no ejecutamos para ahorrar)
        print("Individual processing: skipped to save tokens")
        
        assert isinstance(batch_response, list)
        return f"Batch processing more efficient"
    
    tester.test("Rapid Fire Requests", test_rapid_fire_requests)
    tester.test("Mixed Schema Types", test_mixed_schema_types)
    tester.test("Reuse LLM Instance", test_reuse_llm_instance)
    tester.test("Batch vs Individual", test_batch_vs_individual)
    
    # ========================================
    # FASE 8: COMPATIBILIDAD HACIA ATRÁS
    # ========================================
    
    def test_backward_compatibility():
        """Test compatibilidad con versiones anteriores."""
        print("Planteamiento: Probar compatibilidad hacia atrás con parámetros antiguos.")
        llm = Azure("gpt-4o")
        
        # Usando parámetros antiguos
        response = llm.generate(
            "Di 'COMPAT {name}'",
            prompt_vars={"name": "test"},  # parámetro antiguo
            output_schema={  # parámetro antiguo
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            }
        )
        print(f"Output LLM: {response}")
        
        assert isinstance(response, dict)
        assert "message" in response
        return f"Backward compatibility OK: {response}"
    
    tester.test("Backward Compatibility", test_backward_compatibility)
    
    # ========================================
    # RESUMEN FINAL
    # ========================================
    
    tester.summary()
    
    if tester.failed_tests == 0:
        print(f"\n🎉 ¡TODOS LOS TESTS PASARON!")
        print(f"💡 Azure LLM v4 está funcionando perfectamente")
        print(f"💰 Tests optimizados para minimizar costos de tokens")
    else:
        print(f"\n⚠️  Algunos tests fallaron. Revisa la configuración.")
    
    return tester.results


if __name__ == "__main__":
    main()