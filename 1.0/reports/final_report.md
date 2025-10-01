Título

CardioPredict — Programa de análisis predictivo de enfermedades cardíacas usando Machine Learning

Fundamento 
Este proyecto se enmarca dentro del campo de la Inteligencia Artificial y más específicamente en el área de aprendizaje automático supervisado (supervised learning). En este paradigma, el objetivo es que un modelo aprenda a partir de un conjunto de datos etiquetado, en el cual cada observación está asociada a un resultado conocido (etiqueta). En nuestro caso, cada registro corresponde a un paciente, descrito mediante variables clínicas, y la etiqueta indica la presencia o ausencia de enfermedad cardíaca.

El problema a resolver es de clasificación binaria, ya que la variable objetivo (target) puede tomar dos valores:

0 → paciente sin enfermedad cardíaca,

1 → paciente con enfermedad cardíaca.

El modelo entrenado debe ser capaz de clasificar correctamente nuevos casos basándose en los patrones aprendidos.

Algoritmos de aprendizaje supervisado utilizados

Se consideran distintos algoritmos apropiados para datos clínicos tabulares:

Regresión Logística:
Modelo estadístico ampliamente usado en medicina, que estima la probabilidad de pertenencia a una de las clases a partir de una combinación lineal de los predictores. Es interpretable, lo que permite analizar la influencia de cada variable clínica en el riesgo cardíaco.

Árboles de Decisión y Random Forest:
Los árboles permiten dividir el espacio de características en regiones homogéneas. Los Random Forest (conjunto de múltiples árboles) mejoran la precisión y reducen el riesgo de sobreajuste, además de ofrecer medidas de importancia de variables.

Modelos de ensamble como XGBoost (opcional):
Utilizan boosting para combinar clasificadores débiles y obtener un modelo más robusto y preciso, con gran desempeño en problemas tabulares.

Evaluación del modelo

Para determinar la calidad predictiva, se aplicarán métricas estándar en clasificación:

Exactitud (Accuracy): proporción de casos correctamente clasificados.

Precisión (Precision): fracción de predicciones positivas correctas (fiabilidad del modelo cuando predice enfermedad).

Sensibilidad o Recall: capacidad del modelo para detectar pacientes realmente enfermos (minimizar falsos negativos es crítico en medicina).

Matriz de confusión: representación detallada de aciertos y errores en ambas clases.

Curva ROC y AUC: permiten evaluar el compromiso entre sensibilidad y especificidad en distintos umbrales.

Conjunto de datos utilizado

Se emplearán los datasets cargados (provenientes de UCI y Kaggle), que contienen registros clínicos de pacientes con atributos clave:

age: Edad del paciente.

sex: Sexo del paciente (1 = hombre, 0 = mujer).

cp: Tipo de dolor en el pecho (angina típica, atípica, no anginosa, asintomática).

trestbps: Presión arterial en reposo (mm Hg).

chol: Nivel de colesterol sérico (mg/dl).

fbs: Azúcar en sangre en ayunas (>120 mg/dl).

thalach: Frecuencia cardíaca máxima alcanzada.

target: Variable objetivo (0 = sin enfermedad, 1 = con enfermedad).

Estos atributos fueron seleccionados por su relevancia clínica y porque han sido validados en estudios previos como factores de riesgo para enfermedades cardiovasculares.
Datos

Datasets usados: (coloca los .zip que subiste en data/raw/): por ejemplo:

heart_cleveland.zip → heart_cleveland.csv (UCI Cleveland)

framingham.zip → framingham.csv (Kaggle Framingham)

Atributos clave: age, sex, cp, trestbps, chol, fbs, thalach, target (0/1).

Metodología (resumen de pasos)

Carga y unificación de datasets (si usás más de uno): normalizar nombres de columnas, homogeneizar codificaciones.

EDA: análisis univariado y bivariado, correlaciones y detección de outliers.

Limpieza: imputación de valores faltantes (mediana para numéricos, modo para categóricos), corrección de tipos.

Feature engineering: one-hot encoding para categoricals, generación de features derivadas si aplica (IMC si hay peso/altura, etc.).

Partición: entrenamiento 70% / test 30%, con random_state fijo. Además validación cruzada estratificada (k=5).

Modelado: baseline (LogisticRegression) → RandomForest → XGBoost. Hyperparam tuning con RandomizedSearchCV o GridSearchCV con StratifiedKFold.

Evaluación: accuracy, precision, recall (sensibilidad), F1, ROC-AUC, matriz de confusión.

Interpretabilidad: importancia de variables (feature_importances_ y SHAP).

Entrega: notebooks, best_model.joblib, metrics.csv, PPT y video.mp4.

Limpieza (detalle)

Valores fuera de rango:

trestbps (presión): revisar > 0 y dentro de percentiles 0.5-99.5; truncar o winsorize si es necesario.

chol: igual.

Valores 0 en variables que no pueden ser 0 (algunos datasets usan 0 para missing) — tratarlos como NaN y luego imputar.

Categóricas:

sex: estandarizar (0/1).

cp: mapear a 4 categorías (0..3) y one-hot encode.

Balance de clases:

Revisar proporción de target. Si hay desbalance marcado (>1:3), usar técnicas: class_weight='balanced' en modelos o SMOTE en entrenamiento.

Modelado (detalle técnico)

Pipelines de scikit-learn (ColumnTransformer + Pipeline).

Imputador: SimpleImputer(median/most_frequent).

Escalador: StandardScaler para features numéricos (necesario para SVM/LogReg).

Algoritmos: LogisticRegression (liblinear o saga), RandomForestClassifier, XGBClassifier.

Validación: StratifiedKFold(n_splits=5, shuffle=True, random_state=42).

Métricas: foco en Recall (sensibilidad) y AUC.

Interpretabilidad

SHAP values (TreeExplainer para RandomForest/XGBoost) — gráficos summary, dependence (ej. edad vs probabilidad).

Resultados esperados

Esperar AUC en rangos típicos 0.75-0.9 dependiendo dataset y feature set.

Importantes predictores: edad, cp (tipo de dolor), thalach, chol, trestbps.

Limitaciones

Dataset limitado y sesgado por origen (no representa población total).

Variables clínicas ausentes (ej. IMC, antecedentes familiares) limitan desempeño.

Resultados complementarios a decisión clínica, no sustituyen un diagnóstico.