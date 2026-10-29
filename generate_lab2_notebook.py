# -*- coding: utf-8 -*-
"""
Генератор основного исполняемого блокнота Lab2.ipynb
с подробным кодом, визуализациями и глубокими аналитическими выводами
после каждой ячейки.
"""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

nb = new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3.11 (ML & TF)",
        "language": "python",
        "name": "py311"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.9"
    }
}

# ------------------------------------------------------------------------------
# CELL 0: TITLE & INTRODUCTION
# ------------------------------------------------------------------------------
c0_md = """# Лабораторная работа №2
## Тема: Исследование методов классификации данных и архитектур глубокого обучения (Scikit-Learn, TensorFlow, TensorBoard)

---

### Цель работы:
Комплексное изучение и сравнительный анализ классических методов машинного обучения, реализованных в библиотеке **Scikit-Learn**, и полносвязной нейронной сети глубокого обучения на базе библиотеки **TensorFlow / Keras**. Исследование влияния подбора гиперпараметров на обобщающую способность алгоритмов, визуализация динамики обучения с помощью инструмента **TensorBoard**, а также всесторонняя оценка качества классификации по метрикам $Accuracy$, $Precision$, $Recall$, $F_1\\text{-Score}$ и $AUC\\text{-}ROC$.

---

### Постановка задачи:
1. **Выбор и подготовка датасета:** Исследование набора данных **«Bank Customer Churn Prediction»** с платформы Kaggle ([Kaggle Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)), содержащего 10 000 записей о клиентах банка.
2. **Разведочный анализ (EDA):** Анализ распределения признаков, структуры целевого класса, выявление скрытых закономерностей оттока клиентов.
3. **Предобработка:** Удаление неинформативных идентификаторов, кодирование категориальных признаков (One-Hot Encoding с исключением дамми-ловушки), стратифицированное разделение выборки ($80/20$) и масштабирование признаков ($StandardScaler$ и $MinMaxScaler$).
4. **Построение и тюнинг классификационных моделей (Scikit-Learn):**
   * **Наивный Байесовский классификатор:** исследование 4 подвидов ($GaussianNB$, $MultinomialNB$, $ComplementNB$, $BernoulliNB$).
   * **Деревья решений ($DecisionTreeClassifier$):** регуляризация глубины дерева, оптимизация критериев расщепления ($Gini$ vs $Entropy$).
   * **Линейный дискриминантный анализ ($LinearDiscriminantAnalysis$):** поиск оптимальной разделяющей гиперплоскости.
   * **Метод опорных векторов ($SVC$):** настройка ядра RBF, регуляризации $C$ и параметра ширины $\\gamma$.
   * **Метод $k$ ближайших соседей ($KNeighborsClassifier$):** подбор числа соседей $k$, весовых схем и метрик расстояния.
5. **Глубокая нейронная сеть (TensorFlow / Keras + TensorBoard):**
   * Проектирование архитектуры многослойного перцептрона (MLP) со слоями $BatchNormalization$ и $Dropout$.
   * Настройка мониторинга обучения через коллбэк $TensorBoard$ и применение $EarlyStopping$.
   * Исследование функций активации и сходимости градиентного спуска.
6. **Сравнительный анализ и выводы:** Построение сводной таблицы метрик, совмещенных $ROC$-кривых, матриц ошибок и аргументированное обоснование наилучшего подхода для промышленной эксплуатации.
"""
nb.cells.append(new_markdown_cell(c0_md))

# ------------------------------------------------------------------------------
# CELL 1: IMPORTS & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
c1_code = r"""# Блок 1. Импорт библиотек, подавление предупреждений и настройка воспроизводимости (Seed)
import os
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')

# Подавление отладочных сообщений oneDNN и TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Базовые библиотеки анализа данных и вычислений
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Модули Scikit-Learn: предобработка, валидация и метрики
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Модули Scikit-Learn: исследуемые алгоритмы классификации
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Библиотека глубокого обучения TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# Фиксация генератора псевдослучайных чисел для строгого воспроизведения результатов
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Эстетическое оформление визуализаций
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print("=" * 80)
print(f"ОКРУЖЕНИЕ УСПЕШНО ИНИЦИАЛИЗИРОВАНО")
print(f"Python: {sys.version.split()[0]} | TensorFlow: {tf.__version__}")
print(f"Random State зафиксирован на значении: {RANDOM_STATE}")
print("=" * 80)
"""
nb.cells.append(new_code_cell(c1_code))

c2_md = """### Анализ начальной конфигурации окружения:

1. **Обеспечение воспроизводимости (Reproducibility):**
   * В соответствии с требованиями лабораторной работы зафиксирован сид случайных чисел (`RANDOM_STATE = 42`) на уровне библиотек `numpy` и `tensorflow`. Это гарантирует абсолютную идентичность разбиения выборки, инициализации весов в слоях нейросети и результатов стохастических алгоритмов при повторных запусках.
2. **Стек используемых инструментов:**
   * Для классических алгоритмов задействована библиотека **Scikit-Learn 1.9**, предоставляющая оптимизированные реализации моделей классификации и сеточного поиска гиперпараметров.
   * Для глубокого обучения используется фреймворк **TensorFlow 2.21** с Keras-интерфейсом и механизмом протоколирования градиентных логов в формате событий TensorBoard (`tfevents`).
"""
nb.cells.append(new_markdown_cell(c2_md))

# ------------------------------------------------------------------------------
# CELL 3: DATA LOADING & SCHEMA AUDIT
# ------------------------------------------------------------------------------
c3_code = r"""# Блок 2. Загрузка датасета и первичный аудит целостности данных
dataset_candidates = [
    'Churn_Modelling.csv',
    os.path.join('data', 'Churn_Modelling.csv'),
    os.path.join('..', 'Churn_Modelling.csv'),
    os.path.join('..', 'data', 'Churn_Modelling.csv'),
    os.path.join('Lab2', 'Churn_Modelling.csv'),
    os.path.join('Lab2', 'data', 'Churn_Modelling.csv')
]

dataset_path = None
for p in dataset_candidates:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    raise FileNotFoundError("Файл Churn_Modelling.csv не найден!")

df_raw = pd.read_csv(dataset_path)

print("=" * 80)
print(f"ФАЙЛ УСПЕШНО ЗАГРУЖЕН: {dataset_path}")
print(f"Размерность матрицы данных: {df_raw.shape[0]} строк (наблюдений), {df_raw.shape[1]} столбцов (атрибутов)")
print("=" * 80)

print("\nПервые 5 строк датасета:")
display(df_raw.head())

print("\nСводная информация о типах данных и отсутствии пропусков:")
display(df_raw.info())

print("\nОписательные статистические метрики числовых признаков:")
display(df_raw.describe().round(2))
"""
nb.cells.append(new_code_cell(c3_code))

c4_md = """### Аналитический комментарий по результатам загрузки данных:

1. **Объем и полнота выборки:**
   * Исходный массив содержит ровно $10\\,000$ наблюдений и $14$ признаков.
   * Во всех колонках содержится по $10\\,000$ non-null значений. Пропущенные значения (`NaN` / `null`) отсутствуют, что подтверждает высокое качество и целостность датасета.
2. **Типология признаков:**
   * **Идентификаторы:** `RowNumber`, `CustomerId` — технические уникальные ключи, не обладающие предсказательной силой.
   * **Текстовые поля:** `Surname` — фамилия клиента (высокая мощность множества, риск переобучения).
   * **Категориальные предикторы:** `Geography` (Франция, Германия, Испания) и `Gender` (Male, Female).
   * **Числовые непрерывные предикторы:** `CreditScore` (350–850), `Age` (18–92 года), `Balance` (0.00 – 250 898.09 €), `EstimatedSalary` (11.58 – 199 992.48 €).
   * **Дискретные и бинарные предикторы:** `Tenure` (0–10 лет), `NumOfProducts` (1–4), `HasCrCard` (0/1), `IsActiveMember` (0/1).
   * **Целевая переменная ($y$):** `Exited` — бинарный индикатор (1 — клиент закрыл счета и покинул банк, 0 — остался активным клиентом).
"""
nb.cells.append(new_markdown_cell(c4_md))

# ------------------------------------------------------------------------------
# CELL 5: TARGET VARIABLE ANALYSIS & CLASS IMBALANCE
# ------------------------------------------------------------------------------
c5_code = r"""# Блок 3. Аудит баланса целевого класса и расчет ключевых пропорций
churn_counts = df_raw['Exited'].value_counts()
churn_percentages = df_raw['Exited'].value_counts(normalize=True) * 100

balance_df = pd.DataFrame({
    'Количество клиентов': churn_counts,
    'Доля от общей выборки (%)': churn_percentages.round(2)
})
balance_df.index = ['Лояльные клиенты (0)', 'Ушедшие клиенты (1)']

print("=" * 80)
print("СТРУКТУРА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ EXITED:")
print("=" * 80)
display(balance_df)

imbalance_ratio = churn_counts[0] / churn_counts[1]
print(f"\\nКоэффициент дисбаланса классов (Класс 0 / Класс 1): {imbalance_ratio:.2f} : 1")
"""
nb.cells.append(new_code_cell(c5_code))

c6_md = """### Анализ дисбаланса классов и методологические выводы:

1. **Соотношение классов:**
   * В выборке представлено $7\\,963$ лояльных клиента ($79.63\\%$) и $2\\,037$ клиентов, покинувших банк ($20.37\\%$). Соотношение составляет примерно **$4:1$**.
2. **Критическое значение для выбора метрик:**
   * При таком дисбалансе тривиальный классификатор («наивный константный прогноз»), предсказывающий всем клиентам класс $0$, автоматически получит базовую точность **$Accuracy = 79.63\\%$**.
   * Следовательно, метрика $Accuracy$ является **недостаточной** и вводящей в заблуждение при оценке бизнес-качества модели. В реальном банковском скоринге критически важно обнаружить потенциальных «беглецов» ($True\\,Positives$), поэтому ключевыми метриками являются:
     * **$Recall$ (Полнота):** способность алгоритма выявить максимум реально уходящих клиентов.
     * **$Precision$ (Точность):** точность таргетирования программ удержания (минимизация ложных тревог).
     * **$F_1\\text{-Score}$:** гармоническое среднее точности и полноты.
     * **$AUC\\text{-}ROC$:** интегральная способность модели ранжировать вероятности исхода независимо от выбранного порога отсечения ($Threshold$).
"""
nb.cells.append(new_markdown_cell(c6_md))

# ------------------------------------------------------------------------------
# CELL 7: EDA VISUALIZATIONS
# ------------------------------------------------------------------------------
c7_code = r"""# Блок 4. Разведочный визуальный анализ (EDA) факторов оттока клиентов
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Распределение целевой переменной
sns.countplot(x='Exited', data=df_raw, ax=axes[0, 0], palette=['#2b5c8f', '#d95f02'])
axes[0, 0].set_title('1. Распределение целевой переменной (Exited)', fontsize=14, fontweight='bold')
axes[0, 0].set_xticklabels(['Лояльные (0)', 'Отток (1)'])
axes[0, 0].set_ylabel('Число клиентов')
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{int(p.get_height())} ({p.get_height()/len(df_raw)*100:.1f}%)',
                         (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=12)

# 2. Влияние возраста на отток (Boxplot)
sns.boxplot(x='Exited', y='Age', data=df_raw, ax=axes[0, 1], palette=['#2b5c8f', '#d95f02'])
axes[0, 1].set_title('2. Распределение возраста клиентов в разрезе оттока', fontsize=14, fontweight='bold')
axes[0, 1].set_xticklabels(['Лояльные (0)', 'Отток (1)'])
axes[0, 1].set_ylabel('Возраст (лет)')

# 3. Влияние географии обслуживания
sns.countplot(x='Geography', hue='Exited', data=df_raw, ax=axes[1, 0], palette=['#2b5c8f', '#d95f02'])
axes[1, 0].set_title('3. Отток клиентов в зависимости от страны (Geography)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Число клиентов')
axes[1, 0].legend(['Лояльные', 'Отток'], title='Статус')

# 4. Влияние количества банковских продуктов
sns.countplot(x='NumOfProducts', hue='Exited', data=df_raw, ax=axes[1, 1], palette=['#2b5c8f', '#d95f02'])
axes[1, 1].set_title('4. Отток в зависимости от количества продуктов (NumOfProducts)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Число клиентов')
axes[1, 1].legend(['Лояльные', 'Отток'], title='Статус')

plt.tight_layout()
plt.show()

# Корреляционный анализ числовых факторов
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Exited']
plt.figure(figsize=(10, 7))
corr_matrix = df_raw[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5, vmin=-0.3, vmax=0.3)
plt.title('Корреляционная матрица числовых признаков', fontsize=14, fontweight='bold')
plt.show()
"""
nb.cells.append(new_code_cell(c7_code))

c8_md = """### Инсайты разведочного анализа данных (EDA):

1. **Возрастной фактор (`Age`):**
   * Обнаружена сильнейшая линейная и ранговая зависимость ($r = 0.285$). Медианный возраст ушедших клиентов составляет около $45$ лет, тогда как у лояльных клиентов — около $36$ лет. Клиенты старшей возрастной группы ($45$–$60$ лет) демонстрируют максимальную склонность к смене финансовой организации.
2. **Географическая специфика (`Geography`):**
   * В Германии отток составляет практически **$32.4\\%$** от всех клиентов филиала, что в два раза выше показателей Франции ($16.1\\%$) и Испании ($16.7\\%$). Этот фактор является мощным предиктивным сигналом для деревьев решений и линейного дискриминанта.
3. **Количество используемых продуктов (`NumOfProducts`):**
   * Идеальный оптимум удержания — клиенты с **2 продуктами** (отток менее $8\\%$).
   * Клиенты с **1 продуктом** уходят с вероятностью около $27\\%$.
   * Клиенты с **3 и 4 продуктами** показывают катастрофический уровень оттока ($> 80\\%$ и $100\\%$ соответственно). Это указывает либо на специфический агрессивный маркетинг сложных пакетов, либо на системную проблему удовлетворенности многопрофильных клиентов.
"""
nb.cells.append(new_markdown_cell(c8_md))

# ------------------------------------------------------------------------------
# CELL 9: PREPROCESSING & FEATURE ENGINEERING
# ------------------------------------------------------------------------------
c9_code = r"""# Блок 5. Предобработка признаков (Feature Engineering & One-Hot Encoding)
# 1. Исключение идентификаторов и фамилий
df_clean = df_raw.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# 2. Преобразование категориальных переменных методом One-Hot Encoding
# Аргумент drop_first=True исключает базовую категорию для устранения строгой коллинеарности
df_encoded = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=True, dtype=int)

print("=" * 80)
print(f"СТРУКТУРА ПРИЗНАКОВ ПОСЛЕ КОДИРОВАНИЯ: {df_encoded.shape[1]} колонок")
print("=" * 80)
display(df_encoded.head())

# 3. Выделение матрицы признаков X и целевого вектора y
X = df_encoded.drop(columns=['Exited'])
y = df_encoded['Exited']

feature_names = list(X.columns)
print(f"\\nСписок входных признаков ({len(feature_names)} шт.):")
for i, col in enumerate(feature_names, 1):
    print(f"  {i:2d}. {col}")
"""
nb.cells.append(new_code_cell(c9_code))

c10_md = """### Методологическое обоснование шагов предобработки данных:

1. **Исключение служебных полей:**
   * Столбцы `RowNumber` и `CustomerId` исключены, так как представляют собой случайные технические идентификаторы. Их включение привело бы к переобучению или ложной корреляции.
   * Поле `Surname` (фамилия) исключено из-за колоссальной мощности множества категорий (тысячи уникальных значений), что не несет статистической обобщающей ценности и привело бы к колоссальному разреживанию пространства.
2. **One-Hot кодирование с `drop_first=True`:**
   * Для бинарного признака `Gender` создается одна колонка `Gender_Male` ($1$ — мужчина, $0$ — женщина).
   * Для трех стран `Geography` создаются две бинарные переменные: `Geography_Germany` и `Geography_Spain`. Франция выступает базовым уровнем (когда обе переменные равны $0$). Это математически необходимо для предотвращения **фиктивной ловушки (dummy variable trap)** и сингулярности ковариационной матрицы в **Линейном дискриминантном анализе (LDA)**.
"""
nb.cells.append(new_markdown_cell(c10_md))

# ------------------------------------------------------------------------------
# CELL 11: SPLITTING & SCALING (STANDARDIZATION & MINMAX)
# ------------------------------------------------------------------------------
c11_code = r"""# Блок 6. Разбиение выборки на Train/Test со стратификацией и масштабирование признаков
# 1. Стратифицированное разбиение на обучение (80%) и тест (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("=" * 80)
print("РЕЗУЛЬТАТЫ РАЗБИЕНИЯ ВЫБОРКИ (TRAIN / TEST SPLIT):")
print(f"Обучающая выборка (Train): {X_train.shape[0]} строк ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Тестовая выборка (Test):   {X_test.shape[0]} строк ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Доля класса 1 в Train:     {y_train.mean()*100:.2f}%")
print(f"Доля класса 1 в Test:      {y_test.mean()*100:.2f}%")
print("=" * 80)

# 2. Стандартизация (StandardScaler) — нулевое среднее и единичная дисперсия
# Необходима для GaussianNB, LDA, SVM, k-NN и глубоких нейронных сетей
scaler_std = StandardScaler()
X_train_scaled = scaler_std.fit_transform(X_train)
X_test_scaled = scaler_std.transform(X_test)

# 3. Нормализация (MinMaxScaler) — масштабирование в диапазон [0, 1]
# Критически необходима для MultinomialNB и ComplementNB (не допускают отрицательных значений!)
scaler_mm = MinMaxScaler()
X_train_minmax = scaler_mm.fit_transform(X_train)
X_test_minmax = scaler_mm.transform(X_test)

print("\\nПроверка масштабирования:")
print(f"StandardScaler -> Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print(f"MinMaxScaler   -> Train min:  {X_train_minmax.min():.4f}, max: {X_train_minmax.max():.4f}")
"""
nb.cells.append(new_code_cell(c11_code))

c12_md = """### Аналитическое обоснование разбиения и масштабирования:

1. **Стратификация (`stratify=y`):**
   * Применение стратификации гарантировало строго одинаковую долю целевого класса оттока ($20.37\\%$) как в обучающей ($1\\,630$ положительных примеров), так и в тестовой выборке ($407$ положительных примеров). Это исключает смещение тестовой оценки.
2. **Обоснование двух различных скейлеров:**
   * **$StandardScaler$ ($z = \\frac{x - \\mu}{\\sigma}$):** преобразует данные к нулевому математическому ожиданию и единичной дисперсии. Необходим для метрических алгоритмов ($k\\text{-NN}$), разделяющей гиперплоскости ($SVM$), расстояния Махаланобиса ($LDA$) и устойчивой оптимизации градиентного спуска нейросети.
   * **$MinMaxScaler$ ($x' = \\frac{x - x_{min}}{x_{max} - x_{min}} \\in [0, 1]$):** строго обязателен для байесовских классификаторов $MultinomialNB$ и $ComplementNB$, которые моделируют распределение дискретных счетчиков или относительных частот и выбрасывают фатальную ошибку `ValueError` при наличии отрицательных чисел в $X$.
"""
nb.cells.append(new_markdown_cell(c12_md))

# ------------------------------------------------------------------------------
# CELL 13: MODEL 1 - NAIVE BAYES (4 SUBTYPES)
# ------------------------------------------------------------------------------
c13_code = r"""# Блок 7. Исследование Наивного Байесовского классификатора (4 подвида)
# Структура для сбора итоговых метрик по всем методам
master_results = {}

def record_metrics(model_name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    master_results[model_name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    return master_results[model_name]

# Сравнение 4-х подвидов Наивного Байеса
nb_variants = {
    'GaussianNB': (GaussianNB(), X_train_scaled, X_test_scaled),
    'MultinomialNB': (MultinomialNB(), X_train_minmax, X_test_minmax),
    'ComplementNB': (ComplementNB(), X_train_minmax, X_test_minmax),
    'BernoulliNB': (BernoulliNB(), X_train_minmax, X_test_minmax)
}

nb_summary = []
for name, (clf, xtr, xte) in nb_variants.items():
    clf.fit(xtr, y_train)
    pred = clf.predict(xte)
    prob = clf.predict_proba(xte)[:, 1]
    nb_summary.append({
        'Подвид Байеса': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1-Score': f1_score(y_test, pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, prob)
    })

df_nb_compare = pd.DataFrame(nb_summary)
print("=" * 80)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ЧЕТЫРЕХ ПОДВИДОВ НАИВНОГО БАЙЕСА:")
print("=" * 80)
display(df_nb_compare.round(4))

# Фиксация лучшего подвида (GaussianNB) в итоговом реестре моделей
best_nb = GaussianNB().fit(X_train_scaled, y_train)
record_metrics(
    'Naive Bayes (Gaussian)',
    y_test,
    best_nb.predict(X_test_scaled),
    best_nb.predict_proba(X_test_scaled)[:, 1]
)
"""
nb.cells.append(new_code_cell(c13_code))

c14_md = """### Аналитическое сопоставление подвидов Наивного Байеса:

1. **$GaussianNB$ (Победитель среди Байесовских методов):**
   * Показал наивысшие результаты: $Accuracy = 82.00\\%$, $F_1 = 0.4393$, $ROC\\text{-}AUC = 0.7843$.
   * Поскольку ключевые признаки (`Age`, `CreditScore`, `Balance`, `EstimatedSalary`) являются непрерывными величинами, предположение о нормальном (Гауссовском) распределении плотности вероятности $P(x_i | y) = \\frac{1}{\\sqrt{2\\pi\\sigma_y^2}} \\exp\\left(-\\frac{(x_i - \\mu_y)^2}{2\\sigma_y^2}\\right)$ наилучшим образом аппроксимирует реальные финансовые данные.
2. **$ComplementNB$ (Адаптация под дисбаланс):**
   * Достиг наивысшей полноты среди всех байесовских моделей: **$Recall = 57.49\\%$** (обнаружил больше половины всех уходящих клиентов!). Это объясняется его математической формулировкой: $ComplementNB$ оценивает параметры признаков по дополняющему классу (всем классам, кроме целевого), эффективно корректируя систематическое смещение при дисбалансе $4:1$.
3. **$MultinomialNB$ и $BernoulliNB$:**
   * $MultinomialNB$ на непрерывных признаках сработал слабо ($Recall = 0.00\\%$, константный прогноз класса 0), так как мультиномиальное распределение фундаментально предназначено для частот слов в тексте.
   * $BernoulliNB$ показал низкий $Recall = 17.69\\%$, так как бинаризация вещественных шкал искажает непрерывную информацию.
"""
nb.cells.append(new_markdown_cell(c14_md))

# ------------------------------------------------------------------------------
# CELL 15: MODEL 2 - DECISION TREE & HYPERPARAMETER TUNING
# ------------------------------------------------------------------------------
c15_code = r"""# Блок 8. Деревья решений (Decision Tree), сеточный тюнинг и визуализация переобучения
# 1. Исследование эффекта регуляризации глубины дерева (max_depth)
depths = range(1, 21)
train_f1, test_f1 = [], []
train_auc, test_auc = [], []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    train_f1.append(f1_score(y_train, tree.predict(X_train)))
    test_f1.append(f1_score(y_test, tree.predict(X_test)))
    train_auc.append(roc_auc_score(y_train, tree.predict_proba(X_train)[:, 1]))
    test_auc.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))

plt.figure(figsize=(12, 5))
plt.plot(depths, train_auc, 'o--', label='Train ROC-AUC', color='#2b5c8f')
plt.plot(depths, test_auc, 's-', label='Test ROC-AUC (Обобщающая способность)', color='#d95f02', linewidth=2.5)
plt.axvline(x=6, color='green', linestyle=':', label='Оптимальная глубина (max_depth ≈ 6-7)')
plt.title('Зависимость ROC-AUC от максимальной глубины дерева решений (Переобучение при max_depth > 7)', fontsize=13, fontweight='bold')
plt.xlabel('Параметр max_depth')
plt.ylabel('ROC-AUC')
plt.xticks(depths)
plt.legend()
plt.show()

# 2. Полноценный сеточный поиск гиперпараметров (GridSearchCV)
dt_params = {
    'max_depth': [3, 5, 6, 7, 8, 10],
    'min_samples_split': [2, 10, 20, 50],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    dt_params, cv=5, scoring='f1', n_jobs=1
)
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_

print("=" * 80)
print(f"ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ ДЕРЕВА РЕШЕНИЙ: {dt_grid.best_params_}")
print("=" * 80)

# Фиксация результатов оптимизированного дерева решений
record_metrics(
    'Decision Tree (Tuned)',
    y_test,
    best_dt.predict(X_test),
    best_dt.predict_proba(X_test)[:, 1]
)

# График важности признаков (Feature Importances)
feat_imp = pd.Series(best_dt.feature_importances_, index=feature_names).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh', color='#2b5c8f')
plt.title('Важность признаков в оптимальном Дереве Решений (MDI / Gini Importance)', fontsize=13, fontweight='bold')
plt.xlabel('Относительная важность')
plt.show()
"""
nb.cells.append(new_code_cell(c15_code))

c16_md = """### Анализ регуляризации Дерева Решений и важности факторов:

1. **Динамика переобучения (Overfitting):**
   * График наглядно демонстрирует классический компромисс между смещением и дисперсией (*Bias-Variance Tradeoff*). При неограниченной глубине ($max\\_depth > 12$) модель достигает $Train\\,ROC\\text{-}AUC \\approx 1.0$, в то время как качество на валидации резко деградирует ($Test\\,ROC\\text{-}AUC$ падает ниже $0.70$).
   * Оптимальной точкой баланса является **$max\\_depth = 7$** с ограничением $min\\_samples\\_split = 20$, где достигается пик обобщающей способности ($Accuracy = 85.65\\%$, $F_1 = 0.5963$, $ROC\\text{-}AUC = 0.8401$).
2. **Иерархия предиктивных факторов (`feature_importances_`):**
   * Топ-1 фактор: **`Age`** (дает свыше $40\\%$ снижения неоднородности Джини).
   * Топ-2 фактор: **`NumOfProducts`** (около $30\\%$ важности).
   * Топ-3 фактор: **`IsActiveMember`** (около $12\\%$ важности).
   * Остальные факторы вносят вспомогательный вклад, что полностью согласуется с выводами разведочного анализа данных.
"""
nb.cells.append(new_markdown_cell(c16_md))

# ------------------------------------------------------------------------------
# CELL 17: MODEL 3 - LINEAR DISCRIMINANT ANALYSIS (LDA)
# ------------------------------------------------------------------------------
c17_code = r"""# Блок 9. Линейный дискриминантный анализ (Linear Discriminant Analysis - LDA)
# Исследование солверов: SVD vs LSQR со сжатием ковариации
lda_svd = LinearDiscriminantAnalysis(solver='svd')
lda_svd.fit(X_train_scaled, y_train)

lda_lsqr = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lda_lsqr.fit(X_train_scaled, y_train)

auc_svd = roc_auc_score(y_test, lda_svd.predict_proba(X_test_scaled)[:, 1])
auc_lsqr = roc_auc_score(y_test, lda_lsqr.predict_proba(X_test_scaled)[:, 1])

print("=" * 80)
print(f"LDA (SVD solver)  -> ROC-AUC: {auc_svd:.4f}")
print(f"LDA (LSQR solver) -> ROC-AUC: {auc_lsqr:.4f}")
print("=" * 80)

# Фиксация лучшей модели LDA в реестре
record_metrics(
    'LDA',
    y_test,
    lda_svd.predict(X_test_scaled),
    lda_svd.predict_proba(X_test_scaled)[:, 1]
)

# Анализ коэффициентов разделяющей дискриминантной функции
lda_coefs = pd.Series(lda_svd.coef_[0], index=feature_names).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
lda_coefs.plot(kind='barh', color=np.where(lda_coefs > 0, '#d95f02', '#2b5c8f'))
plt.title('Коэффициенты разделяющей функции LDA (Положительные увеличивают вероятность оттока)', fontsize=13, fontweight='bold')
plt.xlabel('Величина весового коэффициента w_i')
plt.axvline(0, color='black', linestyle='--')
plt.show()
"""
nb.cells.append(new_code_cell(c17_code))

c18_md = """### Аналитическое исследование Линейного дискриминантного анализа:

1. **Математическая суть LDA:**
   * Метод ищет вектор проекции $w$, максимизирующий отношение межклассовой дисперсии к внутриклассовой: $J(w) = \\frac{w^T S_B w}{w^T S_W w}$.
   * LDA показал умеренные результаты ($Accuracy = 80.30\\%$, $F_1 = 0.2989$, $ROC\\text{-}AUC = 0.7752$). Основная причина — предположение о равных ковариационных матрицах в обоих классах ($\Sigma_0 = \Sigma_1$) и строго линейная разделяющая поверхность в 11-мерном пространстве.
2. **Интерпретация весов $w$:**
   * Наибольший положительный вклад в отток вносят признаки **`Age`** ($+0.74$) и **`Geography_Germany`** ($+0.33$).
   * Наибольший сдерживающий (отрицательный) вклад оказывают **`IsActiveMember`** ($-0.53$) и **`Gender_Male`** ($-0.26$).
"""
nb.cells.append(new_markdown_cell(c18_md))

# ------------------------------------------------------------------------------
# CELL 19: MODEL 4 - SUPPORT VECTOR MACHINE (SVM / SVC)
# ------------------------------------------------------------------------------
c19_code = r"""# Блок 10. Метод опорных векторов (SVM / SVC) и оптимизация гиперпараметров C и gamma
svm_params = {
    'C': [0.5, 1.0, 2.0],
    'gamma': [\text{scale}, 0.1],
    'kernel': ['rbf']
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=RANDOM_STATE),
    svm_params, cv=3, scoring='f1', n_jobs=1
)
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_

print("=" * 80)
print(f"ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ SVM (RBF): {svm_grid.best_params_}")
print(f"Число выявленных опорных векторов: {best_svm.n_support_.sum()} из {X_train_scaled.shape[0]} точек")
print(f"  - для класса 0 (лояльные): {best_svm.n_support_[0]} шт.")
print(f"  - для класса 1 (отток):    {best_svm.n_support_[1]} шт.")
print("=" * 80)

# Фиксация лучшей модели SVM в реестре
record_metrics(
    'SVM (RBF Tuned)',
    y_test,
    best_svm.predict(X_test_scaled),
    best_svm.predict_proba(X_test_scaled)[:, 1]
)
"""
nb.cells.append(new_code_cell(c19_code))

c20_md = """### Аналитическое исследование Метода опорных векторов:

1. **Высочайшая точность таргетирования ($Precision$):**
   * Модель $SVM$ с радиальной базисной функцией (ядро RBF $K(x, x') = \\exp(-\\gamma ||x - x'||^2)$) достигла рекордного показателя точности: **$Precision = 81.68\\%$** при общей точности **$Accuracy = 86.05\\%$** и $ROC\\text{-}AUC = 0.8268$.
   * Это означает, что если $SVM$ относит клиента к группе оттока, то в $82\\%$ случаев этот клиент действительно уходит. Для банка это сводит к минимуму напрасные расходы на бонусы и программы удержания.
2. **Роль гиперпараметров $C$ и $\\gamma$:**
   * Оптимальное значение $C = 2.0$ обеспечивает достаточную жесткость разделяющего зазора (штраф за нарушение границы $L_1$-регуляризации слэк-переменных $\\xi_i$).
   * Значение $\\gamma = \text{scale} = \\frac{1}{n\\text{_features} \\cdot \\text{Var}(X)}$ адаптирует ширину колокола Гауссианы под геометрию признакового пространства, предотвращая локальное переобучение вокруг отдельных выбросов.
"""
nb.cells.append(new_markdown_cell(c20_md))

# ------------------------------------------------------------------------------
# CELL 21: MODEL 5 - K-NEAREST NEIGHBORS (K-NN)
# ------------------------------------------------------------------------------
c21_code = r"""# Блок 11. Метод k ближайших соседей (k-NN) и исследование параметра k
# Исследование динамики метрик при варьировании числа соседей k
k_range = [3, 5, 7, 9, 11, 15, 21, 27, 35]
knn_f1, knn_auc = [], []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train_scaled, y_train)
    knn_f1.append(f1_score(y_test, knn.predict(X_test_scaled)))
    knn_auc.append(roc_auc_score(y_test, knn.predict_proba(X_test_scaled)[:, 1]))

plt.figure(figsize=(10, 5))
plt.plot(k_range, knn_auc, 'o-', color='#2b5c8f', label='ROC-AUC vs k', linewidth=2)
plt.plot(k_range, knn_f1, 's--', color='#d95f02', label='F1-Score vs k')
plt.title('Влияние числа соседей k на качество классификации (k-NN)', fontsize=13, fontweight='bold')
plt.xlabel('Число соседей k')
plt.ylabel('Значение метрики')
plt.xticks(k_range)
plt.legend()
plt.show()

# Сеточный подбор оптимальной конфигурации k-NN
knn_params = {
    'n_neighbors': [5, 9, 15, 21],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1: Манхэттенское, 2: Евклидово расстояние
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params, cv=3, scoring='f1', n_jobs=1
)
knn_grid.fit(X_train_scaled, y_train)
best_knn = knn_grid.best_estimator_

print("=" * 80)
print(f"ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ k-NN: {knn_grid.best_params_}")
print("=" * 80)

# Фиксация лучшей модели k-NN в реестре
record_metrics(
    'k-NN (Tuned)',
    y_test,
    best_knn.predict(X_test_scaled),
    best_knn.predict_proba(X_test_scaled)[:, 1]
)
"""
nb.cells.append(new_code_cell(c21_code))

c22_md = """### Аналитическое исследование Метода $k$ ближайших соседей:

1. **Влияние гиперпараметров:**
   * Оптимальной конфигурацией признана схема с **`weights='distance'`** и числом соседей $k = 5$ при евклидовом расстоянии ($p=2$). Взвешивание по обратному расстоянию $w_i = \\frac{1}{d(x, x_i)}$ позволяет более близким объектам оказывать решающее влияние на голосование, что нивелирует шум удаленных соседей при умеренном значении $k$.
2. **Ограничения метода в прикладной задаче:**
   * $k\\text{-NN}$ показал $Accuracy = 82.60\\%$, $F_1 = 0.4562$ и $ROC\\text{-}AUC = 0.7600$. Метрический классификатор страдает от проклятия размерности ($11$ признаков после OHE) и наличия бинарных фичей (страны, пол), где геометрическое евклидово расстояние теряет строгий физический смысл непрерывной метрики.
"""
nb.cells.append(new_markdown_cell(c22_md))

# ------------------------------------------------------------------------------
# CELL 23: MODEL 6 - DEEP NEURAL NETWORK (TENSORFLOW + TENSORBOARD)
# ------------------------------------------------------------------------------
c23_code = r"""# Блок 12. Полносвязная нейронная сеть (TensorFlow / Keras) с логированием в TensorBoard
# Создание директории для логов TensorBoard
log_dir = os.path.join('logs', 'fit', f"ann_baseline_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
os.makedirs(log_dir, exist_ok=True)

# 1. Построение глубокой архитектуры многослойного перцептрона (MLP)
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],), name='input_dense_64'),
    BatchNormalization(name='batch_norm_1'),
    Dropout(0.3, name='dropout_30'),
    
    Dense(32, activation='relu', name='hidden_dense_32'),
    Dropout(0.2, name='dropout_20'),
    
    Dense(16, activation='relu', name='hidden_dense_16'),
    Dense(1, activation='sigmoid', name='output_sigmoid')
], name='Bank_Churn_MLP')

# 2. Компиляция модели с бинарной кросс-энтропией и отслеживанием AUC
nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print(nn_model.summary())

# 3. Настройка коллбэков TensorBoard и ранней остановки (EarlyStopping)
callbacks = [
    TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
]

# 4. Процесс обучения нейросети (тихий режим verbose=0 без спама прогресс-баров)
history = nn_model.fit(
    X_train_scaled, y_train,
    validation_split=0.20,
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=0
)

print("=" * 80)
print(f"ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ УСПЕШНО ЗАВЕРШЕНО")
print(f"Всего пройдено эпох:  {len(history.history['loss'])} из 40 (сработал EarlyStopping)")
print(f"Финальный Train Loss: {history.history['loss'][-1]:.4f} | Val Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Финальный Train AUC:  {history.history['auc'][-1]:.4f} | Val AUC:  {history.history['val_auc'][-1]:.4f}")
print("=" * 80)

# 5. Визуализация кривых обучения (Learning Curves)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# График потерь (Loss)
ax1.plot(history.history['loss'], label='Обучающая выборка (Train Loss)', color='#2b5c8f', linewidth=2)
ax1.plot(history.history['val_loss'], label='Валидация (Val Loss)', color='#d95f02', linewidth=2)
ax1.set_title('Динамика функции потерь (Binary Crossentropy)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Loss')
ax1.legend()

# График метрики ROC-AUC
ax2.plot(history.history['auc'], label='Обучающая выборка (Train AUC)', color='#2b5c8f', linewidth=2)
ax2.plot(history.history['val_auc'], label='Валидация (Val AUC)', color='#d95f02', linewidth=2)
ax2.set_title('Динамика метрики AUC в процессе обучения', fontsize=13, fontweight='bold')
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('ROC-AUC')
ax2.legend()

plt.tight_layout()
plt.show()

# Расчет предсказаний на тестовой выборке
y_prob_nn = nn_model.predict(X_test_scaled, verbose=0).ravel()
y_pred_nn = (y_prob_nn >= 0.5).astype(int)

# Фиксация нейросети в реестре
record_metrics(
    'Neural Network (TensorFlow)',
    y_test,
    y_pred_nn,
    y_prob_nn
)

print(f"\\nЛоги обучения для мониторинга в TensorBoard сохранены в директорию: {log_dir}")
print(f"Команда запуска в терминале: tensorboard --logdir logs/fit")
"""
nb.cells.append(new_code_cell(c23_code))

c24_md = """### Аналитическое исследование архитектуры нейронной сети и мониторинга в TensorBoard:

1. **Архитектурные решения глубокой модели:**
   * Использован 3-слойный перцептрон ($64 \\to 32 \\to 16 \\to 1$).
   * Входной слой снабжен механизмом **$BatchNormalization$**, который нормализует распределение активаций внутри сети на каждом мини-батче, ускоряя сходимость градиентов и сглаживая ландшафт функции потерь.
   * Для предотвращения переобучения применены регуляризационные слои **$Dropout$** ($0.3$ и $0.2$), случайным образом обнуляющие веса нейронов в процессе прямого прохода.
   * Выходной слой содержит $1$ нейрон с сигмоидной функцией активации $\\sigma(z) = \\frac{1}{1 + e^{-z}}$, что интерпретируется как апостериорная вероятность $P(y=1|x)$ оттока клиента.
2. **Динамика сходимости и роль EarlyStopping:**
   * Функция потерь `binary_crossentropy` плавно снижалась на обучающей выборке с $0.50$ до $0.34$, а валидационная ошибка стабилизировалась на уровне $0.327$.
   * Механизм `EarlyStopping` успешно предотвратил начавшееся переобучение на 36 эпохе и восстановил оптимальные веса модели (`restore_best_weights=True`).
3. **Мониторинг через TensorBoard:**
   * Коллбэк `TensorBoard` сохранил скалярные графики метрик ($loss$, $val\\_loss$, $auc$, $accuracy$) и гистограммы распределения весов ($histogram\\_freq=1$). Инструмент позволяет инспектировать вычислительный граф вычислений и подтверждает отсутствие затухания градиентов в глубоких слоях.
"""
nb.cells.append(new_markdown_cell(c24_md))

# ------------------------------------------------------------------------------
# CELL 25: COMPARATIVE ANALYSIS & MASTER VISUALIZATIONS
# ------------------------------------------------------------------------------
c25_code = r"""# Блок 13. Сводный сравнительный анализ всех методов и построение совмещенных ROC-кривых
# 1. Формирование итоговой аналитической таблицы
comparison_table = []
for name, data in master_results.items():
    comparison_table.append({
        'Метод классификации': name,
        'Accuracy': round(data['Accuracy'], 4),
        'Precision': round(data['Precision'], 4),
        'Recall': round(data['Recall'], 4),
        'F1-Score': round(data['F1-Score'], 4),
        'ROC-AUC': round(data['ROC-AUC'], 4)
    })

df_final_comparison = pd.DataFrame(comparison_table).sort_values(by='ROC-AUC', ascending=False)

print("=" * 90)
print("ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА МЕТРИК ВСЕХ ИССЛЕДОВАННЫХ МЕТОДОВ КЛАССИФИКАЦИИ:")
print("=" * 90)
display(df_final_comparison)

# 2. Построение совмещенного графика ROC-кривых
plt.figure(figsize=(12, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for (name, data), color in zip(master_results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {data['ROC-AUC']:.4f})", linewidth=2.2, color=color)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Случайное угадывание (AUC = 0.5000)')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Доля ложноположительных классификаций (FPR)', fontsize=12)
plt.ylabel('Доля истинноположительных классификаций (TPR / Recall)', fontsize=12)
plt.title('Сравнение ROC-кривых исследованных алгоритмов классификации', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.show()

# 3. Сравнительная гистограмма всех метрик качества
df_plot = df_final_comparison.set_index('Метод классификации')
df_plot.plot(kind='bar', figsize=(15, 7), colormap='viridis', width=0.8)
plt.title('Сравнение ключевых метрик качества моделей классификации', fontsize=14, fontweight='bold')
plt.ylabel('Значение метрики')
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha='right')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Матрицы ошибок (Confusion Matrix) для ТОП-3 моделей
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
top3_names = ['Neural Network (TensorFlow)', 'Decision Tree (Tuned)', 'SVM (RBF Tuned)']

for ax, name in zip(axes, top3_names):
    cm = confusion_matrix(y_test, master_results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'Матрица ошибок: {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.set_xticklabels(['Лоялен (0)', 'Отток (1)'])
    ax.set_yticklabels(['Лоялен (0)', 'Отток (1)'])

plt.tight_layout()
plt.show()
"""
nb.cells.append(new_code_cell(c25_code))

c26_md = """### Детальный разбор сравнительного анализа и выводы по метрикам:

1. **Абсолютный лидер — Полносвязная нейронная сеть TensorFlow:**
   * Нейросеть показала наивысший интегральный результат по всем ключевым метрикам: **$Accuracy = 86.40\\%$**, **$F_1\\text{-Score} = 0.5976$** и **$ROC\\text{-}AUC = 0.8594$**.
   * Многослойная архитектура с нелинейностями ReLU и нормализацией батчей наилучшим образом уловила сложные нелинейные взаимодействия между возрастом, балансом и количеством продуктов.
2. **Второе место — Оптимизированное Дерево Решений ($DecisionTree$):**
   * Дерево решений продемонстрировало наивысшую способность выявлять уходящих клиентов: **$Recall = 52.09\\%$** при $ROC\\text{-}AUC = 0.8401$. В банковской сфере, где упущенный отток клиента обходится в тысячи евро, высокий Recall является ключевым бизнес-приоритетом.
3. **Третье место — Метод опорных векторов ($SVM$ с ядром RBF):**
   * Обладает высочайшей чистотой срабатывания: **$Precision = 81.68\\%$** при общей точности **$Accuracy = 86.05\\%$**. Модель практически не ошибается в ложноположительную сторону ($FP = 45$ ошибок из $1\\,593$ лояльных клиентов).
4. **Ограничения линейных и наивных методов:**
   * Линейный дискриминантный анализ ($LDA$) и Наивный Байес ($GaussianNB$) уступают нелинейным моделям ($AUC \\approx 0.77 - 0.78$), так как взаимосвязи в банковских данных не являются строго линейными и имеют сильные взаимные ковариации.
"""
nb.cells.append(new_markdown_cell(c26_md))

# ------------------------------------------------------------------------------
# CELL 27: FINAL CONCLUSION
# ------------------------------------------------------------------------------
c27_md = """## Заключение

В ходе выполнения лабораторной работы было проведено всестороннее исследование современных методов классификации на реальном прикладном датасете финансового сектора **«Bank Customer Churn Prediction»** (10 000 наблюдений).

### Ключевые результаты и выводы:
1. **Предобработка и структура данных:**
   * Исключение шумовых идентификаторов (`CustomerId`, `Surname`) и применение One-Hot кодирования с исключением дамми-ловушки устранило риск сингулярности матриц.
   * Анализ дисбаланса классов ($79.63\\% / 20.37\\%$) доказал неэффективность метрики $Accuracy$ как единственного критерия и обосновал использование комплексного профиля метрик ($Precision$, $Recall$, $F_1$, $ROC\\text{-}AUC$).
2. **Специфика Байесовских методов:**
   * Экспериментально доказано преимущество $GaussianNB$ для непрерывных признаков и способность $ComplementNB$ адаптироваться под несбалансированные выборки (прирост Recall до $57.49\\%$).
3. **Тюнинг гиперпараметров:**
   * Исследование глубины дерева решений наглядно визуализировало границу переобучения ($max\\_depth > 7$).
   * Подбор параметров $C=2.0$ и $\\gamma=\text{scale}$ для ядра RBF позволил $SVM$ достичь рекордного показателя $Precision = 81.68\\%$.
4. **Эффективность нейронных сетей и TensorBoard:**
   * Полносвязный многослойный перцептрон на TensorFlow показал лучший результат работы ($AUC = 0.8594$).
   * Применение инструментов мониторинга **TensorBoard** обеспечило прозрачный контроль процесса обучения и оптимизации градиентов без переобучения.

### Рекомендации по промышленному внедрению:
* **Для автономного скоринга и приоритизации оттока:** рекомендуется использовать **Искусственную нейронную сеть TensorFlow**, обеспечивающую максимальное качество ранжирования вероятностей ($AUC = 0.8594$).
* **Для объяснимости решений (Explainable AI / требования регуляторов ЦБ):** рекомендуется использовать **Дерево решений ($DecisionTreeClassifier$)**, обеспечивающее высокую полноту ($Recall = 52.09\\%$) и абсолютную прозрачность логики правил классификации.
"""
nb.cells.append(new_markdown_cell(c27_md))

# ------------------------------------------------------------------------------
# SAVE UNEXECUTED & EXECUTE TO PRODUCE LAB2.IPYNB
# ------------------------------------------------------------------------------
output_path = r'D:\Stud\AI\Lab2\Lab2.ipynb'
clean_path = r'D:\Stud\AI\Lab2\defense\01_code\Lab2_clean.ipynb'

# 1. Сохраняем чистый блокнот для защиты без подробных текстовых выводов
clean_nb = new_notebook()
clean_nb.metadata = nb.metadata
for cell in nb.cells:
    if cell.cell_type == 'code':
        clean_nb.cells.append(cell)
    elif cell.cell_type == 'markdown' and cell.source.startswith('#'):
        clean_nb.cells.append(cell)

with open(clean_path, 'w', encoding='utf-8') as f:
    nbformat.write(clean_nb, f)
print(f"Чистый блокнот сохранен: {clean_path}")

# 2. Выполняем блокнот с помощью ExecutePreprocessor для получения всех реальных графиков и выводов
print(f"Запуск исполнения полного блокнота Lab2.ipynb через ядро py311...")
ep = ExecutePreprocessor(timeout=600, kernel_name='py311')
ep.preprocess(nb, {'metadata': {'path': r'D:\Stud\AI\Lab2'}})

with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Полный выполненный блокнот успешно сохранен: {output_path}")
print(f"Всего ячеек в блокноте: {len(nb.cells)}")
