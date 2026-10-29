"""
Лабораторная работа №2: Классификация данных и нейронные сети
Тема: Прогнозирование оттока клиентов банка (Bank Customer Churn Prediction)
Методы: Naive Bayes (Gaussian, Multinomial, Complement, Bernoulli),
        Decision Tree, Linear Discriminant Analysis (LDA),
        Support Vector Machines (SVM), k-Nearest Neighbors (k-NN),
        Искусственная нейронная сеть (TensorFlow / Keras + TensorBoard).
"""

import os
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn: модели, разбиение, препроцессинг, метрики
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# TensorFlow / Keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

def main():
    # Воспроизводимость результатов (Random Seed)
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # Настройка визуализации
    sns.set_theme(style='whitegrid', font_scale=1.1)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

    # ==============================================================================
    # 1. ЗАГРУЗКА ДАННЫХ И ПЕРВИЧНЫЙ АУДИТ (EDA)
    # ==============================================================================
    dataset_candidates = [
        'Churn_Modelling.csv',
        os.path.join('data', 'Churn_Modelling.csv'),
        os.path.join('Lab2', 'Churn_Modelling.csv'),
        os.path.join('Lab2', 'data', 'Churn_Modelling.csv'),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Churn_Modelling.csv')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Churn_Modelling.csv'))
    ]

    dataset_path = None
    for p in dataset_candidates:
        if os.path.exists(p):
            dataset_path = p
            break

    if dataset_path is None:
        raise FileNotFoundError("Не удалось найти файл Churn_Modelling.csv ни по одному из стандартных путей!")

    df_raw = pd.read_csv(dataset_path)
    print("=" * 80, flush=True)
    print(f"ДАТАСЕТ УСПЕШНО ЗАГРУЖЕН: {dataset_path}", flush=True)
    print(f"Форма датасета (Shape): {df_raw.shape[0]} строк, {df_raw.shape[1]} столбцов", flush=True)
    print("=" * 80, flush=True)
    print("\nПервые 5 строк исходного датасета:", flush=True)
    print(df_raw.head(), flush=True)

    print("\nПроверка пропущенных значений:", flush=True)
    print(df_raw.isnull().sum(), flush=True)

    print("\nРаспределение целевой переменной Exited:", flush=True)
    print(df_raw['Exited'].value_counts(), flush=True)
    print(df_raw['Exited'].value_counts(normalize=True) * 100, flush=True)

    # ==============================================================================
    # 2. ПРЕДОБРАБОТКА ДАННЫХ (FEATURE ENGINEERING)
    # ==============================================================================
    # Удаление служебных неинформативных идентификаторов
    df = df_raw.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    # One-Hot кодирование категориальных колонок (Geography, Gender)
    # drop_first=True исключает строгую мультиколлинеарность (dummy variable trap)
    df_encoded = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True, dtype=int)

    print("\nФорма датасета после кодирования признаков:", df_encoded.shape, flush=True)
    print("Признаки:", list(df_encoded.columns), flush=True)

    # Разделение на матрицу признаков X и целевой вектор y
    X = df_encoded.drop(columns=['Exited'])
    y = df_encoded['Exited']

    # Разбиение на обучающую (80%) и тестовую (20%) выборки со стратификацией
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nОбучающая выборка: {X_train.shape[0]} объектов", flush=True)
    print(f"Тестовая выборка:   {X_test.shape[0]} объектов", flush=True)

    # Масштабирование признаков:
    # 1) StandardScaler — для GaussianNB, DecisionTree, LDA, SVM, k-NN, Neural Network
    scaler_std = StandardScaler()
    X_train_scaled = scaler_std.fit_transform(X_train)
    X_test_scaled = scaler_std.transform(X_test)

    # 2) MinMaxScaler (0, 1) — специально для MultinomialNB и ComplementNB (требуют X >= 0)
    scaler_mm = MinMaxScaler()
    X_train_minmax = scaler_mm.fit_transform(X_train)
    X_test_minmax = scaler_mm.transform(X_test)

    # Словарь для сохранения результатов
    results = {}

    def evaluate_model(name, y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'y_prob': y_prob
        }
        print(f"\n[{name}]", flush=True)
        print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}", flush=True)
        return results[name]

    # ==============================================================================
    # 3. НАИВНЫЙ БАЙЕСОВСКИЙ КЛАССИФИКАТОР (4 ПОДВИДА)
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("1. НАИВНЫЙ БАЙЕСОВСКИЙ КЛАССИФИКАТОР (Сравнение 4 подвидов)", flush=True)
    print("=" * 80, flush=True)

    nb_models = {
        'GaussianNB': (GaussianNB(), X_train_scaled, X_test_scaled),
        'MultinomialNB': (MultinomialNB(), X_train_minmax, X_test_minmax),
        'ComplementNB': (ComplementNB(), X_train_minmax, X_test_minmax),
        'BernoulliNB': (BernoulliNB(), X_train_minmax, X_test_minmax)
    }

    nb_comparison = {}
    for m_name, (model, xtr, xte) in nb_models.items():
        model.fit(xtr, y_train)
        y_pred = model.predict(xte)
        y_prob = model.predict_proba(xte)[:, 1]
        nb_comparison[m_name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        print(f"  {m_name:15s} -> Acc: {nb_comparison[m_name]['Accuracy']:.4f}, "
              f"Rec: {nb_comparison[m_name]['Recall']:.4f}, "
              f"F1: {nb_comparison[m_name]['F1-Score']:.4f}, "
              f"AUC: {nb_comparison[m_name]['ROC-AUC']:.4f}", flush=True)

    # Для итогового сравнения берем GaussianNB
    best_nb = GaussianNB()
    best_nb.fit(X_train_scaled, y_train)
    evaluate_model('Naive Bayes (Gaussian)', y_test, best_nb.predict(X_test_scaled), best_nb.predict_proba(X_test_scaled)[:, 1])

    # ==============================================================================
    # 4. ДЕРЕВЬЯ РЕШЕНИЙ (DECISION TREE)
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("2. ДЕРЕВЬЯ РЕШЕНИЙ (DECISION TREE)", flush=True)
    print("=" * 80, flush=True)

    dt_params = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 10, 20],
        'criterion': ['gini', 'entropy']
    }

    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        dt_params, cv=3, scoring='f1', n_jobs=1
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    print(f"Лучшие параметры дерева решений: {dt_grid.best_params_}", flush=True)
    evaluate_model('Decision Tree (Tuned)', y_test, best_dt.predict(X_test), best_dt.predict_proba(X_test)[:, 1])

    # ==============================================================================
    # 5. ЛИНЕЙНЫЙ ДИСКРИМИНАНТНЫЙ АНАЛИЗ (LDA)
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("3. ЛИНЕЙНЫЙ ДИСКРИМИНАНТНЫЙ АНАЛИЗ (LDA)", flush=True)
    print("=" * 80, flush=True)

    lda_model = LinearDiscriminantAnalysis(solver='svd')
    lda_model.fit(X_train_scaled, y_train)
    evaluate_model('LDA', y_test, lda_model.predict(X_test_scaled), lda_model.predict_proba(X_test_scaled)[:, 1])

    # ==============================================================================
    # 6. МЕТОД ОПОРНЫХ ВЕКТОРОВ (SVM / SVC)
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("4. МЕТОД ОПОРНЫХ ВЕКТОРОВ (SVM / SVC)", flush=True)
    print("=" * 80, flush=True)

    # Исследуем параметры C и gamma на RBF ядре
    svm_params = {
        'C': [0.5, 1.0, 2.0],
        'gamma': ['scale', 0.1],
        'kernel': ['rbf']
    }

    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_STATE),
        svm_params, cv=3, scoring='f1', n_jobs=1
    )
    svm_grid.fit(X_train_scaled, y_train)
    best_svm = svm_grid.best_estimator_
    print(f"Лучшие параметры SVM: {svm_grid.best_params_}", flush=True)
    evaluate_model('SVM (RBF Tuned)', y_test, best_svm.predict(X_test_scaled), best_svm.predict_proba(X_test_scaled)[:, 1])

    # ==============================================================================
    # 7. МЕТОД k БЛИЖАЙШИХ СОСЕДЕЙ (k-NN)
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("5. МЕТОД k БЛИЖАЙШИХ СОСЕДЕЙ (k-NN)", flush=True)
    print("=" * 80, flush=True)

    knn_params = {
        'n_neighbors': [5, 9, 15, 21],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1: Манхэттенское, 2: Евклидово
    }

    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_params, cv=3, scoring='f1', n_jobs=1
    )
    knn_grid.fit(X_train_scaled, y_train)
    best_knn = knn_grid.best_estimator_
    print(f"Лучшие параметры k-NN: {knn_grid.best_params_}", flush=True)
    evaluate_model('k-NN (Tuned)', y_test, best_knn.predict(X_test_scaled), best_knn.predict_proba(X_test_scaled)[:, 1])

    # ==============================================================================
    # 8. НЕЙРОННАЯ СЕТЬ НА TENSORFLOW / KERAS + TENSORBOARD
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("6. НЕЙРОННАЯ СЕТЬ НА TENSORFLOW / KERAS + TENSORBOARD", flush=True)
    print("=" * 80, flush=True)

    log_base_dir = os.path.join('logs', 'fit')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_nn = os.path.join(log_base_dir, f"nn_baseline_{timestamp}")
    os.makedirs(log_dir_nn, exist_ok=True)

    # Архитектура полносвязной нейронной сети (MLP)
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],), name='dense_1'),
        BatchNormalization(name='batch_norm_1'),
        Dropout(0.3, name='dropout_1'),
        Dense(32, activation='relu', name='dense_2'),
        Dropout(0.2, name='dropout_2'),
        Dense(16, activation='relu', name='dense_3'),
        Dense(1, activation='sigmoid', name='output')
    ])

    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    tb_callback = TensorBoard(log_dir=log_dir_nn, histogram_freq=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(nn_model.summary(), flush=True)

    history = nn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=32,
        callbacks=[tb_callback, early_stop],
        verbose=1
    )

    print(f"Обучение завершено. Эпох пройдено: {len(history.history['loss'])}", flush=True)
    y_prob_nn = nn_model.predict(X_test_scaled, verbose=0).ravel()
    y_pred_nn = (y_prob_nn >= 0.5).astype(int)
    evaluate_model('Neural Network (TensorFlow)', y_test, y_pred_nn, y_prob_nn)

    # ==============================================================================
    # 9. СРАВНИТЕЛЬНЫЙ АНАЛИЗ И СВОДНАЯ ТАБЛИЦА
    # ==============================================================================
    print("\n" + "=" * 80, flush=True)
    print("ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА МЕТРИК", flush=True)
    print("=" * 80, flush=True)

    summary_rows = []
    for name, m in results.items():
        summary_rows.append({
            'Метод классификации': name,
            'Accuracy': round(m['Accuracy'], 4),
            'Precision': round(m['Precision'], 4),
            'Recall': round(m['Recall'], 4),
            'F1-Score': round(m['F1-Score'], 4),
            'ROC-AUC': round(m['ROC-AUC'], 4)
        })

    df_metrics = pd.DataFrame(summary_rows)
    print(df_metrics.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
