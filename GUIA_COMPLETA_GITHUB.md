# 🚀 GUÍA COMPLETA: DESDE GITHUB HASTA LA EJECUCIÓN

## 📋 TABLA DE CONTENIDOS
1. [Preparar el repositorio en GitHub](#1-preparar-github)
2. [Clonar y configurar en tu computadora](#2-clonar-proyecto)
3. [Instalar dependencias](#3-instalar-dependencias)
4. [Ejecutar el código](#4-ejecutar-código)
5. [Solución de problemas](#5-problemas)

---

## 1️⃣ PREPARAR EL REPOSITORIO EN GITHUB

### Paso 1.1: Crear repositorio en GitHub

1. Ve a [github.com](https://github.com) y haz login
2. Click en el botón **"New"** (o ícono +) → **"New repository"**
3. Configura tu repositorio:
   - **Repository name**: `bank-churn-llm`
   - **Description**: `Sistema de predicción de fuga de clientes bancarios usando LLM con LoRA`
   - **Visibility**: ✅ Public (o Private si prefieres)
   - ✅ Add a README file
   - ✅ Add .gitignore → Selecciona **Python**
4. Click en **"Create repository"**

### Paso 1.2: Subir los archivos al repositorio

**Opción A: Usando la interfaz web de GitHub (más fácil)**

1. En tu repositorio, click en **"Add file"** → **"Upload files"**
2. Arrastra estos archivos desde `/mnt/user-data/outputs/`:
   ```
   ✅ churn_prediction_llm.py
   ✅ demo_churn_quick.py
   ✅ churn_benchmark_analysis.ipynb
   ✅ requirements.txt
   ✅ README.md
   ✅ COMO_EJECUTAR.md
   ✅ run_churn_prediction.sh
   ✅ Informe_Churn_Bancario_LLM.docx
   ✅ analisis_roi.png
   ✅ arquitectura_cloud.png
   ✅ comparacion_modelos.png
   ✅ eficiencia_lora.png
   ✅ timeline_implementacion.png
   ```
3. Escribe un mensaje de commit: `"Initial commit - Churn prediction system"`
4. Click en **"Commit changes"**

**Opción B: Usando Git desde terminal (si prefieres CLI)**

```bash
# Desde tu computadora local
cd ~/Documentos  # o donde quieras poner el proyecto

# Clonar el repositorio vacío
git clone https://github.com/TU-USUARIO/bank-churn-llm.git
cd bank-churn-llm

# Copiar todos los archivos del proyecto aquí
# (descárgalos de /mnt/user-data/outputs/ primero)

# Agregar y hacer commit
git add .
git commit -m "Initial commit - Churn prediction system"
git push origin main
```

---

## 2️⃣ CLONAR Y CONFIGURAR EN TU COMPUTADORA

### Paso 2.1: Clonar el repositorio

Abre tu terminal (Terminal en Mac/Linux, CMD o PowerShell en Windows) y ejecuta:

```bash
# Navega a donde quieras guardar el proyecto
cd ~/Documentos  # Mac/Linux
# o
cd C:\Users\TuUsuario\Documentos  # Windows

# Clona el repositorio
git clone https://github.com/TU-USUARIO/bank-churn-llm.git

# Entra al directorio
cd bank-churn-llm

# Verifica que todos los archivos están ahí
ls -la  # Mac/Linux
# o
dir  # Windows
```

### Paso 2.2: Verificar Python instalado

```bash
# Verificar versión de Python (necesitas 3.8 o superior)
python --version
# o si tienes Python 3:
python3 --version

# Deberías ver algo como: Python 3.10.x o superior
```

**Si NO tienes Python instalado:**
- **Windows**: Descarga desde [python.org](https://www.python.org/downloads/)
- **Mac**: `brew install python3` (si tienes Homebrew) o desde [python.org](https://www.python.org/downloads/)
- **Linux**: `sudo apt install python3 python3-pip python3-venv`

---

## 3️⃣ INSTALAR DEPENDENCIAS

### Paso 3.1: Crear entorno virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar el entorno virtual:

# En Mac/Linux:
source venv/bin/activate

# En Windows (CMD):
venv\Scripts\activate.bat

# En Windows (PowerShell):
venv\Scripts\Activate.ps1

# Deberías ver (venv) al inicio de tu línea de comando
```

### Paso 3.2: Instalar dependencias

**Opción A: Instalación rápida (solo para demo)**

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Opción B: Instalación completa (para versión con LLM)**

```bash
pip install --upgrade pip
pip install -r requirements.txt

# O si prefieres instalar manualmente:
pip install torch transformers datasets peft accelerate
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
pip install jupyterlab  # Si quieres usar el notebook
```

**⏱️ Tiempo estimado:**
- Opción A (rápida): ~1-2 minutos
- Opción B (completa): ~5-10 minutos

---

## 4️⃣ EJECUTAR EL CÓDIGO

### 🎯 OPCIÓN 1: DEMO RÁPIDA (RECOMENDADA PARA EMPEZAR)

**Tiempo: 30 segundos | Sin descargas**

```bash
# Asegúrate de tener el entorno virtual activado (debes ver (venv))
python demo_churn_quick.py
```

**✅ Qué verás:**
```
======================================================================
🏦 SISTEMA DE PREDICCIÓN DE FUGA DE CLIENTES BANCARIOS
    Demo Rápida (Sin descarga de modelos LLM)
======================================================================

📊 Generando datos sintéticos de ejemplo...
✅ Datos generados: 5000 registros
📈 Tasa de churn: 26.12%

...

🎯 AUC-ROC Score: 0.9855
📊 Classification Report:
              precision    recall  f1-score   support

    No Churn       0.96      0.94      0.95       739
       Churn       0.84      0.89      0.87       261

...

💰 ANÁLISIS DE ROI
ROI: 577x
```

---

### 🤖 OPCIÓN 2: VERSIÓN COMPLETA CON LLM (DISTILBERT + LORA)

**Tiempo: 10-15 minutos | Descarga ~250MB**

```bash
# Asegúrate de tener TODAS las dependencias instaladas (Opción B del paso 3.2)
python churn_prediction_llm.py
```

**✅ Qué verás:**
```
🤖 Cargando modelo: distilbert-base-uncased
⚡ Aplicando LoRA para fine-tuning eficiente...
trainable params: 38,402 || all params: 66,955,010 || trainable%: 0.0574%

🔄 Tokenizando datasets...
🚀 Iniciando entrenamiento...

Epoch 1/3: 100%|████████████| 250/250 [02:15<00:00]
Epoch 2/3: 100%|████████████| 250/250 [02:12<00:00]
Epoch 3/3: 100%|████████████| 250/250 [02:10<00:00]

✅ Modelo guardado en: ./churn_model_output/

📊 EVALUACIÓN
🎯 AUC-ROC Score: 0.8542

✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE
```

---

### 📊 OPCIÓN 3: JUPYTER NOTEBOOK (ANÁLISIS INTERACTIVO)

```bash
# Instalar JupyterLab (si no lo hiciste)
pip install jupyterlab

# Iniciar Jupyter
jupyter lab

# Se abrirá automáticamente en tu navegador
# Abre el archivo: churn_benchmark_analysis.ipynb
# Ejecuta celda por celda con Shift+Enter
```

---

## 5️⃣ SOLUCIÓN DE PROBLEMAS COMUNES

### ❌ Error: "command not found: python"

**Solución:**
```bash
# Intenta con python3
python3 --version

# Si funciona, usa python3 en lugar de python en todos los comandos
alias python=python3  # Para tu sesión actual
```

### ❌ Error: "No module named 'torch'"

**Solución:**
```bash
# Asegúrate de tener el entorno virtual activado
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Instala torch
pip install torch transformers
```

### ❌ Error: "Permission denied" en Windows PowerShell

**Solución:**
```powershell
# Ejecuta PowerShell como Administrador y corre:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Luego intenta activar de nuevo:
venv\Scripts\Activate.ps1
```

### ❌ Error: "CUDA not available" o warnings de GPU

**Solución:** 
✅ **Esto NO es un error crítico!** El código funcionará en CPU, solo será un poco más lento.

Si quieres usar GPU (opcional):
- Necesitas una GPU NVIDIA compatible
- Instala PyTorch con CUDA: [pytorch.org](https://pytorch.org/get-started/locally/)

### ❌ El script se cuelga o tarda mucho

**Solución:**
```python
# Edita el archivo y reduce el tamaño del dataset
# En demo_churn_quick.py o churn_prediction_llm.py
# Cambia la línea:
n_samples = 5000
# Por:
n_samples = 1000  # Más rápido para probar
```

### ❌ Error de memoria (MemoryError)

**Solución:**
- Cierra otros programas
- Reduce `n_samples` a 1000 o 500
- Reduce `per_device_train_batch_size` a 8 en el script LLM

---

## 📁 ESTRUCTURA FINAL DEL PROYECTO

Después de ejecutar, deberías tener:

```
bank-churn-llm/
├── venv/                          # Entorno virtual (NO subir a Git)
├── churn_model_output/            # Modelo entrenado (generado)
│   ├── config.json
│   ├── model.safetensors
│   └── adapter_config.json
├── metrics.json                   # Métricas (generado)
├── churn_prediction_llm.py        # Script principal LLM
├── demo_churn_quick.py            # Demo rápida
├── churn_benchmark_analysis.ipynb # Notebook
├── requirements.txt               # Dependencias
├── README.md                      # Documentación
├── COMO_EJECUTAR.md              # Esta guía
├── Informe_Churn_Bancario_LLM.docx # Informe
└── *.png                          # Visualizaciones
```

---

## ✅ CHECKLIST DE VERIFICACIÓN

Antes de ejecutar, verifica:

- [ ] Python 3.8+ instalado (`python --version`)
- [ ] Repositorio clonado (`cd bank-churn-llm`)
- [ ] Entorno virtual creado (`python3 -m venv venv`)
- [ ] Entorno virtual activado (ves `(venv)` en terminal)
- [ ] Dependencias instaladas (`pip install pandas numpy scikit-learn`)
- [ ] Archivos del proyecto presentes (`ls -la`)

---

## 🎯 RESUMEN DE COMANDOS (COPIA Y PEGA)

### Setup completo en una sola secuencia:

**Mac/Linux:**
```bash
# 1. Clonar
git clone https://github.com/TU-USUARIO/bank-churn-llm.git
cd bank-churn-llm

# 2. Crear y activar entorno
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn

# 4. Ejecutar demo rápida
python demo_churn_quick.py

# 5. (Opcional) Instalar todo para versión LLM
pip install torch transformers datasets peft accelerate tqdm

# 6. (Opcional) Ejecutar versión completa
python churn_prediction_llm.py
```

**Windows:**
```cmd
REM 1. Clonar
git clone https://github.com/TU-USUARIO/bank-churn-llm.git
cd bank-churn-llm

REM 2. Crear y activar entorno
python -m venv venv
venv\Scripts\activate.bat

REM 3. Instalar dependencias
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn

REM 4. Ejecutar demo rápida
python demo_churn_quick.py

REM 5. (Opcional) Instalar todo para versión LLM
pip install torch transformers datasets peft accelerate tqdm

REM 6. (Opcional) Ejecutar versión completa
python churn_prediction_llm.py
```

---

## 🎓 PARA EL TALLER / ENTREGA

### Lo que debes entregar:

1. **Link al repositorio GitHub**: `https://github.com/TU-USUARIO/bank-churn-llm`
2. **Informe DOCX**: Descarga `Informe_Churn_Bancario_LLM.docx`
3. **Demostración**: Screenshot o video de la ejecución exitosa

### Cómo hacer un buen screenshot:

```bash
# Ejecuta el demo
python demo_churn_quick.py > output.txt

# Ahora tienes todo el output en output.txt
# Abre output.txt y toma screenshot de las métricas
```

---

## 🚀 ¡LISTO PARA COMENZAR!

Si seguiste todos los pasos, ahora puedes:

1. ✅ Ejecutar la demo rápida en 30 segundos
2. ✅ Entrenar el modelo LLM completo en 15 minutos
3. ✅ Analizar resultados en Jupyter Notebook
4. ✅ Presentar tu proyecto con el informe DOCX

**¿Dudas?** Revisa la sección de [Solución de Problemas](#5-problemas) arriba.

---

**¡Éxito con tu proyecto! 🎉**
