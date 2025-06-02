## 🐍 Activar entorno virtual

Para activar el entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate     # En Linux/Mac
.\.venv\Scripts\activate      # En Windows
```

---

## 📦 Instalar librería local `ease_llm`

1. **Verifica que `requirements.txt` incluya:**

   ```txt
   -e ./libs/ease_llm
   ```

2. **Instala las dependencias desde la raíz del proyecto:**

   ```bash
   pip install -r requirements.txt
   ```

Esto instalará `ease_llm` en modo editable y todas las dependencias necesarias.
