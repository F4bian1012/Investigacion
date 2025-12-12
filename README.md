# Investigación TinyML - Arduino Portenta H7

Este proyecto sigue una arquitectura **MLOps** rigurosa adaptada para Sistemas Embebidos (TinyML). El objetivo es garantizar la reproducibilidad de los experimentos, la trazabilidad de los modelos y una clara separación entre el entrenamiento (Python) y el despliegue (C++).

## 📂 Filosofía de la Estructura de Directorios

La organización de carpetas ha sido diseñada siguiendo las mejores prácticas de la industria:

### 1. `data/` - El ciclo de vida de los datos
Separamos los datos en tres estados inmutables para evitar corrupción y fugas de información:
- **`raw/`**: Datos crudos tal como se obtuvieron del sensor o fuente. **Nunca se sobreescriben.**
- **`processed/`**: Datos limpios, normalizados y listos para entrar al modelo.
- **`augmented/`**: Versiones generadas artificialmente para robustecer el entrenamiento (ruido añadido, cambios de tono, etc.).

### 2. `models/` - Gestión de Artefactos
- **`checkpoints/`**: Guardamos los modelos "pesados" de entrenamiento (formato `.keras` o `.h5`). Estos contienen el estado completo del optimizador.
- **`tflite/`**: Aquí residen únicamente los modelos cuantizados y optimizados para el microcontrolador (`.tflite`) y sus conversiones a arreglos de C (`model.h`).

### 3. `src/` vs `deployment/`
- **`src/`**: Contiene todo el código Python para la "ciencia" del proyecto (entrenamiento, evaluación, scripts de utilidad). Es el entorno del Data Scientist.
- **`deployment/arduino_project/`**: Contiene el código fuente C++ final que se cargará en la Portenta H7. Es el entorno del Ingeniero Embebido. Mantener esto separado evita conflictos de dependencias entre Python y C++.

### 4. `logs/` y `config/`
- **`logs/`**: Almacena gráficas de pérdidas (loss curves), métricas de precisión y registros de experimentos. Fundamental para comparar qué iteración del modelo funciona mejor.
- **`config/`**: Archivos de configuración o hiperparámetros.

## 🚀 Cómo empezar

1. Instala las dependencias de Python:
   ```bash
   pip install -r requirements.txt
   ```
2. Realiza tus experimentos en `notebooks/` o scripts en `src/`.
3. Una vez tengas un modelo entrenado, expórtalo a `models/tflite/`.
4. Copia el arreglo C generado a `deployment/arduino_project/` para compilarlo en Arduino IDE.
