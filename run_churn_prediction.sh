#!/bin/bash

echo "🚀 CONFIGURANDO ENTORNO PARA CHURN PREDICTION LLM"
echo "=================================================="

# Activar entorno virtual
source /home/claude/venv_churn/bin/activate

echo "✅ Entorno virtual activado"
echo ""

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip --quiet

# Instalar dependencias (versión ligera para demo rápida)
echo "📦 Instalando dependencias necesarias..."
echo "   (Esto puede tardar 2-3 minutos)"
echo ""

pip install --quiet \
    torch \
    transformers \
    datasets \
    peft \
    accelerate \
    scikit-learn \
    pandas \
    numpy \
    tqdm

echo ""
echo "✅ Todas las dependencias instaladas correctamente"
echo ""
echo "=================================================="
echo "🏃 EJECUTANDO CHURN PREDICTION LLM"
echo "=================================================="
echo ""

# Ejecutar el script principal
python /home/claude/churn_prediction_llm.py

echo ""
echo "=================================================="
echo "✅ EJECUCIÓN COMPLETADA"
echo "=================================================="
