# 🚀 GUÍA DE EJECUCIÓN - Churn Prediction LLM

## 📋 Tienes 2 opciones para ejecutar el proyecto:

---

## ⚡ OPCIÓN 1: DEMO RÁPIDA (Recomendada para empezar)
**Tiempo: ~30 segundos | Sin descargas pesadas**

### Pasos:

```bash
# 1. Activar entorno virtual
source /home/claude/venv_churn/bin/activate

# 2. Instalar dependencias básicas (si no están instaladas)
pip install pandas numpy scikit-learn

# 3. Ejecutar demo rápida
python /home/claude/demo_churn_quick.py
```

✅ **Qué hace:**
- Genera datos sintéticos de 5,000 clientes
- Entrena 2 modelos (Random Forest + Gradient Boosting)
- Muestra métricas (AUC-ROC, Precision, Recall)
- Calcula ROI completo
- Hace predicción de ejemplo

❌ **NO incluye:**
- Fine-tuning de LLM (para eso usa Opción 2)

---

## 🤖 OPCIÓN 2: VERSIÓN COMPLETA CON LLM (DistilBERT + LoRA)
**Tiempo: ~15 minutos | Descarga ~250MB**

### Pasos:

```bash
# 1. Activar entorno virtual
source /home/claude/venv_churn/bin/activate

# 2. Instalar TODAS las dependencias
pip install torch transformers datasets peft accelerate scikit-learn pandas numpy tqdm

# 3. Ejecutar script completo
python /home/claude/churn_prediction_llm.py
```

✅ **Qué hace:**
- Descarga DistilBERT-base-uncased (~250MB)
- Aplica LoRA fine-tuning (solo 0.03% parámetros entrenables)
- Entrena modelo en datos de churn
- Guarda modelo en `./churn_model_output/`
- Genera métricas en `metrics.json`

⚠️ **Requisitos:**
- Conexión a internet (para descargar modelo)
- ~2GB RAM disponible
- 10-15 minutos de tiempo

---

## 🎯 SCRIPT AUTOMÁTICO (TODO EN UNO)

Si prefieres que todo se instale y ejecute automáticamente:

```bash
bash /home/claude/run_churn_prediction.sh
```

Este script:
1. ✅ Activa el entorno virtual
2. ✅ Instala todas las dependencias
3. ✅ Ejecuta el script completo con LLM

---

## 📊 ARCHIVOS GENERADOS

Después de ejecutar, encontrarás:

```
/home/claude/
├── churn_model_output/          # Modelo entrenado (Opción 2)
│   ├── config.json
│   ├── model.safetensors
│   └── adapter_config.json
├── metrics.json                 # Métricas de evaluación
└── venv_churn/                  # Entorno virtual
```

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### Error: "ModuleNotFoundError"
```bash
# Asegúrate de tener el entorno virtual activado
source /home/claude/venv_churn/bin/activate

# Reinstala dependencias
pip install -r /home/claude/requirements.txt
```

### Error: "CUDA not available"
**No es problema!** El script funciona en CPU también, solo será un poco más lento.

### Error de memoria
Si tienes poco RAM, reduce el tamaño del dataset en el script:
```python
# En demo_churn_quick.py o churn_prediction_llm.py
n_samples = 1000  # En lugar de 5000
```

---

## 💡 RECOMENDACIÓN

**Para primera ejecución:** Usa la **OPCIÓN 1** (demo rápida) para ver resultados inmediatamente.

**Para el taller/entrega:** Usa la **OPCIÓN 2** (versión completa) para mostrar el fine-tuning real con LLM.

---

## ✅ VERIFICAR QUE TODO FUNCIONA

Ejecuta este test rápido:

```bash
source /home/claude/venv_churn/bin/activate
python -c "import pandas; import numpy; import sklearn; print('✅ Todo OK!')"
```

Si ves "✅ Todo OK!", estás listo para ejecutar cualquier versión.

---

## 📞 ¿NECESITAS AYUDA?

Si tienes problemas, verifica:
1. ✅ Entorno virtual activado (debes ver `(venv_churn)` en tu terminal)
2. ✅ Dependencias instaladas (`pip list`)
3. ✅ Suficiente espacio en disco (~2GB libre)

---

**¡Listo para ejecutar! 🚀**
