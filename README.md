# 🧠 SignTalkModel  
Modelo de reconocimiento de lenguaje de señas con Python, MediaPipe y TensorFlow

Este proyecto implementa un sistema de reconocimiento de señas utilizando visión por computadora y un modelo LSTM entrenado con puntos clave capturados mediante MediaPipe Holistic. El repositorio contiene herramientas para recolectar datos, entrenar el modelo, probarlo y ejecutarlo en tiempo real.

---

## 📌 Características principales
- Extracción de keypoints de manos, rostro y cuerpo usando MediaPipe Holistic.  
- Modelo LSTM diseñado para reconocer secuencias de gestos.  
- Scripts modulares para recolección de datos, entrenamiento e inferencia.  
- Modelos preentrenados incluidos (`.h5` y `.keras`).  
- Ejecución en tiempo real mediante webcam.  
- Archivo HTML sencillo para pruebas en navegador.

---

## 📂 Estructura del proyecto
│── app.py # Ejecución del modelo en tiempo real
│── collect_key_points.py # Recolección de keypoints para el dataset
│── execute_model.py # Inferencia y pruebas del modelo
│── train_model.py # Entrenamiento del modelo LSTM
│── sign_model.keras # Modelo entrenado
│── SLD_val_acc.weights.h5 # Pesos (mejor accuracy)
│── SLD_val_loss.weights.h5 # Pesos (mejor pérdida)
│── quickstart.py # Demo rápida de uso
│── index.html # Prueba visual en navegador
│── requirements.txt # Dependencias
│── .gitignore

---

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/EddieReynosoR/SignTalkModel.git
cd SignTalkModel
```

### 2. Crear entorno virtual
```bash
python -m venv venv
venv\Scripts\activate  
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 🎥 Recolección de datos
```bash
python collect_key_points.py
```

### 🏋️‍♂️ Entrenamiento del modelo
```bash
python train_model.py
```

### 🤖 Ejecución en tiempo real
```bash
python app.py
```

## 📦 Dependencias principales

- TensorFlow / Keras  
- MediaPipe  
- OpenCV  
- NumPy  
- Flask (si se usa API local)

## 🧩 Mejoras futuras

- Migración del modelo a TensorFlow.js para uso completamente web.  
- Versión móvil para Android/iOS.  
- Dataset más amplio y robusto.  
- Optimización del modelo para dispositivos de bajo rendimiento.
