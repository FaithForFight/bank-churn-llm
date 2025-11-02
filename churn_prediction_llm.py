"""
Sistema de Predicción de Fuga de Clientes Bancarios usando LLM Fine-tuned
Taller Individual - Tópicos Avanzados en IA
Universidad Adolfo Ibáñez
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import json
import warnings
warnings.filterwarnings('ignore')

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Dispositivo: {device}")

class ChurnPredictorLLM:
    """
    Clase principal para predicción de churn usando LLM con LoRA fine-tuning
    """
    
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Inicializa el predictor con el modelo base
        
        Args:
            model_name: Nombre del modelo HuggingFace a usar
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, filepath=None):
        """
        Carga y prepara los datos de churn bancario
        
        Args:
            filepath: Ruta al archivo CSV (opcional, se puede generar sintético)
        """
        if filepath:
            df = pd.read_csv(filepath)
        else:
            # Generar datos sintéticos para demostración
            print("📊 Generando datos sintéticos de ejemplo...")
            np.random.seed(42)
            n_samples = 5000
            
            df = pd.DataFrame({
                'CreditScore': np.random.randint(300, 850, n_samples),
                'Age': np.random.randint(18, 80, n_samples),
                'Tenure': np.random.randint(0, 10, n_samples),
                'Balance': np.random.uniform(0, 250000, n_samples),
                'NumOfProducts': np.random.randint(1, 5, n_samples),
                'HasCrCard': np.random.randint(0, 2, n_samples),
                'IsActiveMember': np.random.randint(0, 2, n_samples),
                'EstimatedSalary': np.random.uniform(10000, 200000, n_samples),
                'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
                'Gender': np.random.choice(['Male', 'Female'], n_samples)
            })
            
            # Generar target con lógica realista
            churn_prob = (
                (df['Age'] > 50).astype(int) * 0.2 +
                (df['Balance'] < 50000).astype(int) * 0.15 +
                (df['NumOfProducts'] < 2).astype(int) * 0.2 +
                (df['IsActiveMember'] == 0).astype(int) * 0.25 +
                (df['Tenure'] < 2).astype(int) * 0.15 +
                np.random.uniform(0, 0.1, n_samples)
            )
            df['Exited'] = (churn_prob > 0.5).astype(int)
        
        print(f"✅ Datos cargados: {len(df)} registros")
        print(f"📈 Tasa de churn: {df['Exited'].mean():.2%}")
        
        return df
    
    def create_text_prompts(self, df):
        """
        Convierte datos estructurados a prompts de texto para el LLM
        
        Args:
            df: DataFrame con los datos
        
        Returns:
            Lista de prompts de texto
        """
        prompts = []
        
        for _, row in df.iterrows():
            prompt = f"""Analiza el siguiente perfil de cliente bancario y determina el riesgo de fuga:
Cliente de {row['Age']} años, {row['Gender']}, ubicado en {row['Geography']}.
Puntaje crediticio: {row['CreditScore']}, Balance: ${row['Balance']:.2f}.
Productos contratados: {row['NumOfProducts']}, Antigüedad: {row['Tenure']} años.
Tarjeta de crédito: {'Sí' if row['HasCrCard'] == 1 else 'No'}, 
Miembro activo: {'Sí' if row['IsActiveMember'] == 1 else 'No'}.
Salario estimado: ${row['EstimatedSalary']:.2f}.

¿Este cliente tiene alto riesgo de abandonar el banco?"""
            
            prompts.append(prompt)
        
        return prompts
    
    def prepare_dataset(self, df, test_size=0.2):
        """
        Prepara el dataset en formato HuggingFace
        
        Args:
            df: DataFrame con los datos
            test_size: Proporción del conjunto de prueba
        
        Returns:
            train_dataset, test_dataset
        """
        # Crear prompts de texto
        texts = self.create_text_prompts(df)
        labels = df['Exited'].values
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Crear datasets de HuggingFace
        train_dataset = Dataset.from_dict({
            'text': X_train,
            'label': y_train
        })
        
        test_dataset = Dataset.from_dict({
            'text': X_test,
            'label': y_test
        })
        
        return train_dataset, test_dataset
    
    def setup_model(self, use_lora=True):
        """
        Configura el modelo con o sin LoRA
        
        Args:
            use_lora: Si True, aplica LoRA para fine-tuning eficiente
        """
        print(f"🤖 Cargando modelo: {self.model_name}")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Cargar modelo base
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        if use_lora:
            print("⚡ Aplicando LoRA para fine-tuning eficiente...")
            
            # Configuración LoRA optimizada para clasificación
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,  # Rank de las matrices LoRA
                lora_alpha=32,  # Factor de escalado
                lora_dropout=0.1,
                target_modules=["q_lin", "v_lin"],  # Para DistilBERT
                bias="none"
            )
            
            # Aplicar LoRA al modelo
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        self.model = model.to(device)
        return self.model
    
    def tokenize_function(self, examples):
        """
        Función de tokenización para el dataset
        """
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    def train(self, train_dataset, test_dataset, output_dir="./churn_model_output"):
        """
        Entrena el modelo con los datos
        
        Args:
            train_dataset: Dataset de entrenamiento
            test_dataset: Dataset de prueba
            output_dir: Directorio para guardar el modelo
        """
        # Tokenizar datasets
        print("🔄 Tokenizando datasets...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        test_dataset = test_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Configuración de entrenamiento optimizada
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=3e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            warmup_steps=100,
            fp16=torch.cuda.is_available(),  # Usar precisión mixta si hay GPU
            push_to_hub=False,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Crear Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Entrenar
        print("🚀 Iniciando entrenamiento...")
        trainer.train()
        
        # Guardar el modelo
        trainer.save_model(output_dir)
        print(f"✅ Modelo guardado en: {output_dir}")
        
        return trainer
    
    def evaluate(self, test_dataset):
        """
        Evalúa el modelo en el conjunto de prueba
        
        Args:
            test_dataset: Dataset de prueba
        
        Returns:
            Diccionario con métricas
        """
        print("📊 Evaluando modelo...")
        
        # Preparar datos para evaluación
        test_texts = test_dataset['text']
        test_labels = test_dataset['label']
        
        # Tokenizar
        inputs = self.tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Predicciones
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        
        # Calcular métricas
        auc_score = roc_auc_score(test_labels, probabilities)
        report = classification_report(test_labels, predictions, target_names=['No Churn', 'Churn'])
        cm = confusion_matrix(test_labels, predictions)
        
        print("\n" + "="*60)
        print("📈 RESULTADOS DE EVALUACIÓN")
        print("="*60)
        print(f"\n🎯 AUC-ROC Score: {auc_score:.4f}")
        print(f"\n📊 Classification Report:\n{report}")
        print(f"\n🔍 Confusion Matrix:")
        print(f"                Predicted")
        print(f"                No Churn  Churn")
        print(f"Actual No Churn  {cm[0][0]:6d}   {cm[0][1]:5d}")
        print(f"       Churn     {cm[1][0]:6d}   {cm[1][1]:5d}")
        print("="*60)
        
        metrics = {
            'auc_roc': auc_score,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return metrics
    
    def predict_single(self, customer_text):
        """
        Realiza predicción para un solo cliente
        
        Args:
            customer_text: Texto descriptivo del cliente
        
        Returns:
            Diccionario con predicción y probabilidad
        """
        self.model.eval()
        
        inputs = self.tokenizer(
            customer_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            prediction = torch.argmax(logits, dim=-1).item()
        
        result = {
            'prediction': 'CHURN' if prediction == 1 else 'NO CHURN',
            'churn_probability': float(probabilities[1]),
            'confidence': float(max(probabilities))
        }
        
        return result


def main():
    """
    Función principal de ejecución
    """
    print("="*70)
    print("🏦 SISTEMA DE PREDICCIÓN DE FUGA DE CLIENTES BANCARIOS")
    print("    Usando LLM Fine-tuning con LoRA")
    print("="*70)
    
    # Inicializar predictor
    predictor = ChurnPredictorLLM(model_name="distilbert-base-uncased")
    
    # Cargar datos
    df = predictor.load_and_prepare_data()
    
    # Preparar datasets
    train_dataset, test_dataset = predictor.prepare_dataset(df)
    print(f"✅ Train dataset: {len(train_dataset)} ejemplos")
    print(f"✅ Test dataset: {len(test_dataset)} ejemplos")
    
    # Configurar modelo con LoRA
    predictor.setup_model(use_lora=True)
    
    # Entrenar
    trainer = predictor.train(train_dataset, test_dataset)
    
    # Evaluar
    metrics = predictor.evaluate(test_dataset)
    
    # Guardar métricas
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\n✅ Métricas guardadas en metrics.json")
    
    # Ejemplo de predicción individual
    print("\n" + "="*70)
    print("🔮 EJEMPLO DE PREDICCIÓN INDIVIDUAL")
    print("="*70)
    
    example_text = """Analiza el siguiente perfil de cliente bancario y determina el riesgo de fuga:
Cliente de 45 años, Male, ubicado en France.
Puntaje crediticio: 650, Balance: $120000.00.
Productos contratados: 2, Antigüedad: 3 años.
Tarjeta de crédito: Sí, 
Miembro activo: No.
Salario estimado: $75000.00.

¿Este cliente tiene alto riesgo de abandonar el banco?"""
    
    prediction = predictor.predict_single(example_text)
    print(f"\n📊 Resultado: {prediction['prediction']}")
    print(f"📈 Probabilidad de churn: {prediction['churn_probability']:.2%}")
    print(f"🎯 Confianza: {prediction['confidence']:.2%}")
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*70)


if __name__ == "__main__":
    main()
