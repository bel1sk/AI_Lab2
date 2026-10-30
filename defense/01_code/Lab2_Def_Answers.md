# Защита лабораторной работы №2: Ответы на вопросы по коду

В данном руководстве собраны исчерпывающие ответы на все **36 контрольных вопросов по программному коду** лабораторной работы №2.
Для удобства и максимальной скорости ответа на защите для каждого вопроса указаны точные координаты в двух представлениях:
1. 📄 **В скрипте Python:** [`defense/01_code/lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py) (с номерами строк);
2. 📓 **В блокноте Jupyter:** [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb) и [`defense/01_code/Lab2_clean.ipynb`](file:///D:/Stud/AI/Lab2/defense/01_code/Lab2_clean.ipynb) (с номерами ячеек и смысловых блоков).

---

## 📌 Сводная таблица быстрой навигации по коду

| № | Контрольный вопрос преподавателя | Строки в [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py) | Ячейка [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb) | Ключевая функция / фрагмент |
|:---:|:---|:---:|:---:|:---|
| **1** | Загрузка датасета, размерность, типы | [**74–80**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L74-L80) | Ячейка 3 | `pd.read_csv()`, `df_raw.shape`, `head()` |
| **2** | Формирование матрицы $X$ и вектора $y$ | [**93–105**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L93-L105) | Ячейка 9 | `X = df_encoded.drop(...)`, `y = df_encoded[...]` |
| **3** | Проверка баланса классов | [**85–87**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L85-L87) | Ячейка 5 | `df_raw['Exited'].value_counts()` |
| **4** | Поиск и обработка пропусков | [**82–83**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L82-L83) | Ячейка 3 | `df_raw.isnull().sum()` |
| **5** | Кодирование категориальных колонок (OHE) | [**95–98**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L95-L98) | Ячейка 9 | `pd.get_dummies(..., drop_first=True)` |
| **6** | Разбиение train/test, random_state, stratify | [**106–113**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L106-L113) | Ячейка 11 | `train_test_split(..., test_size=0.2, stratify=y)` |
| **7** | Масштабирование (fit vs transform) | [**114–124**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L114-L124) | Ячейка 11 | `scaler_std.fit_transform(X_train)`, `transform(X_test)` |
| **8** | Наивный Байес (выбор GaussianNB) | [**153–180**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L153-L180) | Ячейка 13 | `best_nb = GaussianNB()`, 4 подвида |
| **9** | Параметры Байеса (var_smoothing) | [**154, 178**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L154) | Ячейка 13 | `GaussianNB(var_smoothing=1e-9)` |
| **10** | Дерево решений (параметры split, depth) | [**189–200**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L189-L200) | Ячейка 15 | `DecisionTreeClassifier()`, `dt_params` |
| **11** | Глубина, листья и важность признаков | [**200–202**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L200-L202) | Ячейка 15 | `best_dt.get_depth()`, `feature_importances_` |
| **12** | Линейный дискриминантный анализ (LDA) | [**211–214**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L211-L214) | Ячейка 17 | `LinearDiscriminantAnalysis(solver='svd')` |
| **13** | Метод опорных векторов SVM (ядро, C, gamma) | [**223–237**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L223-L237) | Ячейка 19 | `SVC(kernel='rbf', C=..., gamma=...)` |
| **14** | Метод ближайших соседей k-NN (k, weights, p) | [**245–259**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L245-L259) | Ячейка 21 | `KNeighborsClassifier()`, `knn_params` |
| **15** | Вызовы fit и predict для 5 методов | [**162, 199, 212, 233, 255**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L162) | Ячейки 13–21 | `model.fit(X_tr, y_tr)`, `model.predict(X_te)` |
| **16** | Расчет Accuracy | [**128–129, 166**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L128-L129) | Ячейка 13 | `acc = accuracy_score(y_true, y_pred)` |
| **17** | Расчет Precision, Recall, F1 (average) | [**130–132, 167**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L130-L132) | Ячейка 13 | `precision_score`, `recall_score`, `f1_score` |
| **18** | Определение положительного класса (pos_label) | [**104, 140, 164**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L104) | Ячейки 9, 13 | `y = df['Exited']`, `predict_proba(...)[:, 1]` |
| **19** | Матрица ошибок (Confusion Matrix) | [**31, 128–144**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L31) | Ячейка 25 | `confusion_matrix(y_test, y_pred)` |
| **20** | Вычисление AUC-ROC (вероятности) | [**133, 170**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L133) | Ячейка 13 | `auc = roc_auc_score(y_true, y_prob)` |
| **21** | Оценки для ROC-кривой (SVM Platt Scaling) | [**31, 230, 236**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L230) | Ячейки 19, 25 | `SVC(probability=True)`, `roc_curve` |
| **22** | Сводная таблица и единый тестовый сет | [**311–327**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L311-L327) | Ячейка 25 | `df_metrics = pd.DataFrame(summary_rows)` |
| **23** | Пространства поиска гиперпараметров | [**153, 189, 211, 223, 245**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L189-L193) | Ячейки 13–21 | `dt_params`, `svm_params`, `knn_params` |
| **24** | Инструмент подбора (GridSearchCV, циклы) | [**195, 229, 251**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L195-L199) | Ячейки 15, 19, 21 | `GridSearchCV(...)` |
| **25** | Кросс-валидация (cv, scoring) | [**197, 231, 253**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L197) | Ячейки 15, 19, 21 | `cv=3` (или `cv=5`), `scoring='f1'` |
| **26** | Извлечение лучших параметров и модели | [**200–202, 234, 256**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L200-L202) | Ячейки 15, 19, 21 | `best_params_`, `best_estimator_` |
| **27** | График влияния гиперпараметра (depth, k) | [**189–202, 245–258**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L189-L202) | Ячейки 15, 21 | Цикл `for d in depths: ...`, `plt.plot()` |
| **28** | Создание нейросети (Sequential, слои) | [**272–281**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L272-L281) | Ячейка 23 | `nn_model = Sequential([...])` |
| **29** | Размерность, скрытые слои, model.summary | [**274–280, 292**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L274-L280) | Ячейка 23 | `input_shape=(11,)`, `nn_model.summary()` |
| **30** | Функции активации (ReLU, Sigmoid на выходе) | [**274–280**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L274-L280) | Ячейка 23 | `relu`, `Dense(1, activation='sigmoid')` |
| **31** | Вызов compile (loss, optimizer, metrics) | [**283–287**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L283-L287) | Ячейка 23 | `Adam(lr=0.001)`, `binary_crossentropy` |
| **32** | Настройка lr, batch_size, epochs | [**284, 297–298**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L284) | Ячейка 23 | `learning_rate=0.001`, `batch_size=32`, `epochs=40` |
| **33** | Вызов fit (обучение, валидация, history) | [**294–301**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L294-L301) | Ячейка 23 | `history = nn_model.fit(..., validation_split=0.2)` |
| **34** | Предсказания нейросети, порог 0.5 | [**304–306**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L304-L306) | Ячейка 23 | `y_pred_nn = (y_prob_nn >= 0.5).astype(int)` |
| **35** | Сравнение конфигураций нейросети | [**267–270, 289, 311**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L267-L270) | Ячейка 23 | Логирование по папкам в `logs/fit/` |
| **36** | Коллбэк TensorBoard, EarlyStopping | [**267–270, 289–291, 299**](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L289-L291) | Ячейка 23 | `TensorBoard(log_dir=...)`, `EarlyStopping(...)` |

---

## Подробный разбор каждого вопроса для защиты

---

### 1. Где загружается датасет? Как проверить его размер, названия признаков и типы данных?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L74-L80):** строки 74–80.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 3 (Блок 2).
* **Фрагмент кода:**
  ```python
  df_raw = pd.read_csv(dataset_path)
  print(f"Форма датасета (Shape): {df_raw.shape[0]} строк, {df_raw.shape[1]} столбцов")
  print(df_raw.head())
  ```
* **Как проверить параметры:**
  * **Размер (форма):** `df_raw.shape` $\to$ выводит `(10000, 14)` (10 000 строк, 14 колонок).
  * **Названия признаков:** `df_raw.columns.tolist()`.
  * **Типы данных и заполненность:** `df_raw.info()` или `df_raw.dtypes` (числовые `int64`, `float64`, строковые `object`).

---

### 2. Где в коде формируются матрица признаков X и целевая переменная y? Какие столбцы в них входят?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L93-L105):** строки 93–105.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 9 (Блок 5).
* **Фрагмент кода:**
  ```python
  # Удаление неинформативных идентификаторов
  df = df_raw.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
  # OHE-кодирование категориальных признаков
  df_encoded = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True, dtype=int)
  # Формирование матрицы признаков и вектора меток
  X = df_encoded.drop(columns=['Exited'])
  y = df_encoded['Exited']
  ```
* **Что входит в $X$ (11 признаков):**
  1. `CreditScore` (числовой, кредитный рейтинг 350–850)
  2. `Age` (числовой, возраст 18–92)
  3. `Tenure` (дискретный, стаж клиента в банке 0–10 лет)
  4. `Balance` (вещественный, остаток средств на счете)
  5. `NumOfProducts` (дискретный, количество продуктов 1–4)
  6. `HasCrCard` (бинарный, 0/1)
  7. `IsActiveMember` (бинарный, 0/1)
  8. `EstimatedSalary` (вещественный, расчетный доход)
  9. `Geography_Germany` (бинарный OHE)
  10. `Geography_Spain` (бинарный OHE)
  11. `Gender_Male` (бинарный OHE, 1 — мужчина, 0 — женщина)
* **Что входит в $y$:** целевой вектор `Exited` ($0$ — клиент остался, $1$ — ушел из банка).

---

### 3. Покажите, как проверяется количество объектов каждого класса. Какой результат даёт эта проверка на вашем датасете?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L85-L87):** строки 85–87.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 5 (Блок 3).
* **Фрагмент кода:**
  ```python
  print(df_raw['Exited'].value_counts())
  print(df_raw['Exited'].value_counts(normalize=True) * 100)
  ```
* **Результат на датасете:**
  * Класс 0 (лояльные клиенты): **7 963 объекта** ($79.63\%$).
  * Класс 1 (отток клиентов): **2 037 объектов** ($20.37\%$).
  * Соотношение классов составляет примерно **$4:1$** (умеренный дисбаланс).

---

### 4. Где выполняются поиск и обработка пропущенных значений? Что произойдёт с вашим кодом, если в данных появится пропуск?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L82-L83):** строки 82–83.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 3 (Блок 2).
* **Фрагмент кода:**
  ```python
  print("Проверка пропущенных значений:")
  print(df_raw.isnull().sum())
  ```
* **Результат:** Пропусков в датасете нет (`0` по всем 14 колонкам).
* **Что произойдет при появлении пропуска:**
  * Классификаторы Scikit-Learn (`GaussianNB`, `DecisionTree`, `LDA`, `SVC`, `k-NN`) и TensorFlow выбросят исключение: `ValueError: Input contains NaN`.
  * **Решение:** Требуется применить `SimpleImputer` (замена медианой/модой), `KNNImputer` либо удалить строки через `df.dropna()`.

---

### 5. Покажите, как категориальные признаки и метки классов преобразуются в числовой формат. Как восстановить исходное название класса по предсказанию?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L95-L98):** строки 95–98.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 9 (Блок 5).
* **Фрагмент кода:**
  ```python
  df_encoded = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True, dtype=int)
  ```
  * `Gender`: преобразован в `Gender_Male` ($1$ — мужчина, $0$ — женщина).
  * `Geography`: преобразован в `Geography_Germany` и `Geography_Spain` (Франция выступает базовым классом $0, 0$).
* **Восстановление названия класса по предсказанию:**
  * Целевая метка бинарная: $y \in \{0, 1\}$.
  * Восстановление выполняется по словарю маппинга:
    ```python
    target_names = {0: 'Лояльный клиент', 1: 'Отток (расторжение договора)'}
    pred_names = [target_names[pred] for pred in y_pred]
    ```

---

### 6. Где данные разделяются на обучающую и тестовую выборки? Как заданы размер тестовой выборки, random_state и stratify?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L106-L113):** строки 106–113.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 11 (Блок 6).
* **Фрагмент кода:**
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
  )
  ```
* **Параметры:**
  * `test_size=0.20`: 20% отдано под тест ($2\,000$ строк), 80% под обучение ($8\,000$ строк).
  * `random_state=42`: фиксация генератора псевдослучайных чисел для строго воспроизводимого разбиения.
  * `stratify=y`: принудительное сохранение исходной пропорции классов ($79.63\% / 20.37\%$) в обеих выборках.

---

### 7. Покажите, где выполняется масштабирование признаков. На каких данных вызывается fit, а на каких — transform? Почему выбран такой порядок?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L114-L124):** строки 114–124.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 11 (Блок 6).
* **Фрагмент кода:**
  ```python
  # StandardScaler (для GaussianNB, DecisionTree, LDA, SVM, k-NN, Neural Network)
  scaler_std = StandardScaler()
  X_train_scaled = scaler_std.fit_transform(X_train)
  X_test_scaled = scaler_std.transform(X_test)

  # MinMaxScaler (специально для MultinomialNB и ComplementNB, требующих X >= 0)
  scaler_mm = MinMaxScaler()
  X_train_minmax = scaler_mm.fit_transform(X_train)
  X_test_minmax = scaler_mm.transform(X_test)
  ```
* **На каких данных вызывается:**
  * `fit_transform` — строго на **обучающей** выборке (`X_train`).
  * `transform` — на **тестовой** выборке (`X_test`).
* **Почему выбран такой порядок:**
  * Исключение **утечки данных (Data Leakage)**: тестовая выборка должна симулировать реальные новые данные, статистика которых ($\mu$ и $\sigma$) заранее неизвестна. Вычисление параметров масштабирования только по `X_train` гарантирует объективность оценки.

---

### 8. Покажите создание наивного байесовского классификатора. Какой вариант вы выбрали и почему он подходит для ваших признаков?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L153-L180):** строки 153–180 (сравнение 4 моделей, выбор GaussianNB на строках 178–180).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 13 (Блок 7).
* **Фрагмент кода:**
  ```python
  best_nb = GaussianNB()
  best_nb.fit(X_train_scaled, y_train)
  evaluate_model('Naive Bayes (Gaussian)', y_test, 
                 best_nb.predict(X_test_scaled), 
                 best_nb.predict_proba(X_test_scaled)[:, 1])
  ```
* **Почему выбран $GaussianNB$:**
  * Признаки банковских клиентов (`CreditScore`, `Age`, `Balance`, `EstimatedSalary`) являются непрерывными вещественными величинами, моделируемыми нормальным распределением $\mathcal{N}(\mu, \sigma^2)$.
  * Подвиды `MultinomialNB` и `ComplementNB` рассчитаны на дискретные частоты слов (NLP) и на непрерывных данных показали нулевую полноту ($Recall = 0.00\%$).

---

### 9. Где задаются параметры выбранного байесовского классификатора? Покажите изменение одного из них и объясните, что оно меняет.
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L154):** строка 154 и строка 178.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 13 (Блок 7).
* **Фрагмент кода:**
  ```python
  # По умолчанию: var_smoothing=1e-9
  best_nb = GaussianNB(var_smoothing=1e-9)
  ```
* **Что меняет параметр:**
  * `var_smoothing` — коэффициент сглаживания дисперсии (доля от максимальной дисперсии признака), добавляемый к дисперсии для предотвращения деления на ноль.
  * **Пример изменения:** `GaussianNB(var_smoothing=1e-2)` увеличивает искусственную дисперсию, делая кривую Гаусса шире и устойчивее к единичным аномальным выбросам, но слегка сглаживает различия между классами.

---

### 10. Покажите создание дерева решений. Где заданы критерий разбиения, максимальная глубина и ограничения на количество объектов в узлах?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L189-L200):** строки 189–200.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 15 (Блок 8).
* **Фрагмент кода:**
  ```python
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
  ```
  * `criterion`: функция оценки качества расщепления узла (`'gini'` — индекс Джини, `'entropy'` — прирост информации Шеннона).
  * `max_depth`: ограничение глубины дерева (лучшее значение: $7$).
  * `min_samples_split`: минимальное число объектов для разбиения внутреннего узла (лучшее: $20$).

---

### 11. Как в вашем коде получить глубину обученного дерева, количество листьев и важности признаков? Покажите результат.
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L200-L202):** строки 200–202.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 15 (Блок 8).
* **Фрагмент кода:**
  ```python
  depth = best_dt.get_depth()                # глубина (7)
  n_leaves = best_dt.get_n_leaves()          # количество листовых вершин
  importances = best_dt.feature_importances_  # массив важностей признаков
  ```
* **Результат важности признаков:**
  1. `Age`: **$42.1\%$** (ключевой фактор оттока)
  2. `NumOfProducts`: **$31.8\%$** (влияние числа активных банковских продуктов)
  3. `IsActiveMember`: **$11.4\%$** (активность клиента)
  4. `Balance`: **$5.2\%$** (размер депозита)

---

### 12. Где создаётся модель линейного дискриминантного анализа? Какие параметры вы задаёте явно, а какие оставляете по умолчанию?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L211-L214):** строки 211–214.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 17 (Блок 9).
* **Фрагмент кода:**
  ```python
  lda_model = LinearDiscriminantAnalysis(solver='svd')
  lda_model.fit(X_train_scaled, y_train)
  ```
* **Явно задан:** `solver='svd'` (метод сингулярного разложения матрицы $X$, не требующий обращения ковариационной матрицы и устойчивый к коллинеарности).
* **По умолчанию:** `shrinkage=None` (без регуляризации Ledoit-Wolf), `priors=None` (априорные вероятности вычисляются по долям классов $79.63\%$ и $20.37\%$), `tol=1e-4`.

---

### 13. Покажите создание SVM. Где выбирается ядро и задаются параметры C и, если он используется, gamma?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L223-L237):** строки 223–237.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 19 (Блок 10).
* **Фрагмент кода:**
  ```python
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
  ```
  * `kernel='rbf'`: нелинейное Гауссово ядро $\exp(-\gamma \|x - x'\|^2)$.
  * `C=2.0`: регуляризатор (штраф за ошибки классификации).
  * `gamma='scale'`: коэффициент ядра $\gamma = \frac{1}{n_{\text{features}} \cdot \mathrm{Var}(X)}$.

---

### 14. Где создаётся классификатор ближайших соседей? Как задаются количество соседей, способ взвешивания и метрика расстояния?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L245-L259):** строки 245–259.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 21 (Блок 11).
* **Фрагмент кода:**
  ```python
  knn_params = {
      'n_neighbors': [5, 9, 15, 21],
      'weights': ['uniform', 'distance'],
      'p': [1, 2]  # 1: Манхэттенское, 2: Евклидово
  }
  knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, scoring='f1', n_jobs=1)
  knn_grid.fit(X_train_scaled, y_train)
  best_knn = knn_grid.best_estimator_
  ```
  * `n_neighbors`: число голосующих соседей (лучшее: $5$).
  * `weights='distance'`: вес соседа обратно пропорционален расстоянию $w_i = \frac{1}{d(x, x_i)}$.
  * `p=2`: метрика Минковского $p=2$ (классическое Евклидово расстояние).

---

### 15. Покажите вызовы fit и predict для всех пяти методов. Какие данные передаются в каждый вызов и что возвращается?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L162-L164):** строки 162–164, 179–180 (NB), строки 199, 202 (DT), строки 212–213 (LDA), строки 233, 236 (SVM), строки 255, 258 (k-NN).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейки 13, 15, 17, 19, 21.
* **Фрагмент кода:**
  ```python
  # 1. Обучение: передается матрица train и вектор меток train. Возвращается обученная модель.
  model.fit(X_train_scaled, y_train)

  # 2. Предсказание меток: передается матрица признаков test. Возвращается массив классов {0, 1}.
  y_pred = model.predict(X_test_scaled)

  # 3. Предсказание вероятностей: возвращается матрица вероятностей для классов [P(y=0), P(y=1)].
  y_prob = model.predict_proba(X_test_scaled)[:, 1]
  ```

---

### 16. Где вычисляется accuracy? Покажите, какие предсказания и истинные метки используются при расчёте.
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L128-L129):** строки 128–129 (в `evaluate_model`), строка 166.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 13 (Блок 7).
* **Фрагмент кода:**
  ```python
  acc = accuracy_score(y_true, y_pred)
  ```
  * `y_true`: истинные метки из отложенной тестовой выборки `y_test` (размер 2 000).
  * `y_pred`: бинарные дискретные прогнозы классификатора на `X_test_scaled`.

---

### 17. Покажите расчёт precision, recall и F1. Как задан параметр average и почему вы выбрали именно это значение?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L130-L132):** строки 130–132 (в `evaluate_model`), строки 167–169.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 13 (Блок 7).
* **Фрагмент кода:**
  ```python
  prec = precision_score(y_true, y_pred, zero_division=0)
  rec = recall_score(y_true, y_pred, zero_division=0)
  f1 = f1_score(y_true, y_pred, zero_division=0)
  ```
* **Параметр `average`:**
  * По умолчанию равен `average='binary'`.
  * **Почему:** Задача строго бинарная ($0$ и $1$). Значение `'binary'` вычисляет метрики целенаправленно для положительного класса $1$ (клиенты из группы риска оттока). Многоклассовые параметры `'macro'`/`'weighted'` здесь не требуются.

---

### 18. Как в коде определяется положительный класс для бинарной классификации либо порядок классов для многоклассовой задачи?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L104):** строка 104 (`y = df_encoded['Exited']`), строки 140, 164, 180, 202, 213, 236, 258 (срез `[:, 1]`).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейки 9, 13.
* **Ответ:**
  * В Scikit-Learn положительным классом по умолчанию считается метка **`1`** (`pos_label=1`).
  * Порядок классов зафиксирован в атрибуте `model.classes_` (в нашем датасете `array([0, 1])`), поэтому колонка с индексом 1 в `predict_proba(...)[:, 1]` строго соответствует вероятности положительного класса ($Exited = 1$).

---

### 19. Покажите построение матрицы ошибок. Выберите одну ячейку и объясните, какие объекты в неё попали.
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L31):** строка 31 (импорт `confusion_matrix`), строки 128–144, 236.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 25 (Блок 13).
* **Фрагмент кода:**
  ```python
  cm = confusion_matrix(y_test, y_pred)
  ```
* **Разбор ячеек на примере SVM ($N = 2\,000$):**
  * $[0, 0]$ (**$TN = 1548$**): истинно лояльные клиенты, которых модель верно предсказала как оставшихся.
  * $[0, 1]$ (**$FP = 45$**): **ложная тревога**. Клиент лоялен, но модель ошибочно предсказала отток.
  * $[1, 0]$ (**$FN = 242$**): **пропуск цели**. Клиент закрыл счет, а модель не заметила угрозы.
  * $[1, 1]$ (**$TP = 165$**): истинный отток. Модель вовремя выявила уходящего клиента.

---

### 20. Где вычисляется AUC-ROC? Покажите, передаются ли в расчёт вероятности, значения решающей функции или готовые метки классов.
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L133):** строка 133 (`auc = roc_auc_score(y_true, y_prob)`), строка 170.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 13 (Блок 7).
* **Фрагмент кода:**
  ```python
  auc = roc_auc_score(y_true, y_prob)
  ```
* **Что передается:** В расчет передаются **непрерывные вероятности `y_prob`** ($P(y=1|x) \in [0, 1]$), извлеченные из `predict_proba()[:, 1]`.
* **Почему нельзя передавать готовые метки:** При передаче меток $0$ и $1$ кривая вырождается в ломаную из одной точки, и интеграл площади теряет информативность о ранжирующей способности модели при вариации порога отсечения.

---

### 21. Покажите, как получены оценки для построения ROC-кривой каждого классификатора. Как это реализовано для SVM?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L31):** строка 31 (импорт `roc_curve`), строка 230 (`SVC(probability=True)`), строки 236, 304.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 25 (Блок 13).
* **Фрагмент кода:**
  ```python
  fpr, tpr, thresholds = roc_curve(y_test, y_prob)
  ```
* **Реализация для SVM:**
  * По умолчанию SVM строит только разделяющую гиперплоскость и возвращает вещественный отступ $f(x)$.
  * Для получения вероятностей задан параметр **`SVC(probability=True)`**. Внутри Scikit-Learn запускается **калибровка Платта (Platt Scaling)**: поверх отступов обучается логистическая регрессия через 5-кратную кросс-валидацию, формируя калиброванные вероятности `predict_proba()`.

---

### 22. Где результаты всех моделей объединяются в сравнительную таблицу или график? Как проверить, что модели оценивались на одной тестовой выборке?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L311-L327):** строки 311–327 (сводная таблица `df_metrics` из словаря `results`), строки 180, 202, 213, 236, 258, 306.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 25 (Блок 13).
* **Фрагмент кода:**
  ```python
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
  print(df_metrics.to_string(index=False))
  ```
* **Как проверить идентичность выборки:**
  * Во всех вызовах `evaluate_model` первым аргументом передается неизменный массив `y_test` длиной ровно $2\,000$ объектов, а признаки берутся из соответствующего тестового массива (`X_test_scaled` или `X_test_minmax`), полученного из единого разбиения `train_test_split(random_state=42)`.

---

### 23. Покажите пространства поиска гиперпараметров для каждого из пяти методов. Какие значения вы исследовали?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L153-L158):** строки 153–158 (NB), 189–193 (DT), 211 (LDA), 223–227 (SVM), 245–249 (k-NN).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейки 13, 15, 17, 19, 21.
* **Исследованные пространства:**
  1. **Naive Bayes:** 4 архитектуры распределения: `GaussianNB`, `MultinomialNB`, `ComplementNB`, `BernoulliNB`.
  2. **Decision Tree:** `max_depth: [3, 5, 7, 10]`, `min_samples_split: [2, 10, 20]`, `criterion: ['gini', 'entropy']`.
  3. **LDA:** солвер `solver: ['svd', 'lsqr']`.
  4. **SVM:** `C: [0.5, 1.0, 2.0]`, `gamma: ['scale', 0.1]`, `kernel: ['rbf']`.
  5. **k-NN:** `n_neighbors: [5, 9, 15, 21]`, `weights: ['uniform', 'distance']`, `p: [1, 2]`.

---

### 24. Где выполняется подбор гиперпараметров? Покажите используемый инструмент либо самостоятельно написанный цикл перебора.
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L195-L199):** строки 195–199 (DT), строки 229–233 (SVM), строки 251–255 (k-NN).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейки 15, 19, 21.
* **Фрагмент кода:**
  ```python
  dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=3, scoring='f1', n_jobs=1)
  svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=3, scoring='f1', n_jobs=1)
  knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, scoring='f1', n_jobs=1)
  ```
* **Самостоятельные циклы перебора в блокноте:** В ячейке 15 блокнота реализован цикл `for d in range(1, 21):` для исследования переобучения дерева, а в ячейке 21 — цикл перебора числа соседей $k \in [3, 35]$.

---

### 25. Как в коде организована кросс-валидация? Где заданы количество разбиений и метрика выбора лучшей модели?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L197):** строки 197, 231, 253.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейки 15, 19, 21.
* **Фрагмент кода:**
  ```python
  GridSearchCV(estimator=..., param_grid=..., cv=3, scoring='f1', n_jobs=1)
  ```
  * `cv=3`: 3-кратная стратифицированная кросс-валидация (`StratifiedKFold` встроена по умолчанию для задач классификации).
  * `scoring='f1'`: целевая метрика оптимизации (балансирует Precision и Recall на несбалансированном классе $Exited=1$).

---

### 26. Покажите, как из результатов подбора извлекаются лучшие параметры и обученная модель. Где выполняется её итоговая проверка на тестовой выборке?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L200-L202):** строки 200–202 (DT), строки 234–236 (SVM), строки 256–258 (k-NN).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейки 15, 19, 21.
* **Фрагмент кода:**
  ```python
  # 1. Извлечение параметров и лучшей обученной модели
  best_params = dt_grid.best_params_
  best_dt = dt_grid.best_estimator_

  # 2. Итоговая проверка на тесте
  y_pred = best_dt.predict(X_test)
  y_prob = best_dt.predict_proba(X_test)[:, 1]
  evaluate_model('Decision Tree (Tuned)', y_test, y_pred, y_prob)
  ```

---

### 27. Покажите таблицу или график влияния одного гиперпараметра на качество. Какие строки кода формируют данные для этого исследования?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L189-L202):** строки 189–202 (сетка глубины дерева `dt_params`), строки 245–258 (сетка `k` в `knn_params`).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 15 (график `max_depth`), Ячейка 21 (график `k` в k-NN).
* **Фрагмент кода из блокнота:**
  ```python
  depths = range(1, 21)
  train_auc, test_auc = [], []
  for d in depths:
      t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
      train_auc.append(roc_auc_score(y_train, t.predict_proba(X_train)[:, 1]))
      test_auc.append(roc_auc_score(y_test, t.predict_proba(X_test)[:, 1]))
  plt.plot(depths, train_auc, label='Train AUC')
  plt.plot(depths, test_auc, label='Test AUC')
  ```
* **Вывод графика:** При $max\_depth > 7$ начинается переобучение: обучающая точность растет до $1.0$, а тестовая деградирует (с $0.84$ до $0.68$).

---

### 28. Покажите, где создаётся нейронная сеть. Как используется Sequential либо другой выбранный вами способ построения модели?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L272-L281):** строки 272–281.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 23 (Блок 12).
* **Фрагмент кода:**
  ```python
  nn_model = Sequential([
      Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],), name='dense_1'),
      BatchNormalization(name='batch_norm_1'),
      Dropout(0.3, name='dropout_1'),
      Dense(32, activation='relu', name='dense_2'),
      Dropout(0.2, name='dropout_2'),
      Dense(16, activation='relu', name='dense_3'),
      Dense(1, activation='sigmoid', name='output')
  ])
  ```
* **Способ построения:** `Sequential` организует линейный стек слоев, последовательно преобразующих вектор признаков.

---

### 29. Где задаются входная размерность, количество скрытых слоёв и число нейронов в слоях Dense? Покажите вывод model.summary().
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L274-L280):** строки 274–280 (конфигурация), строка 292 (`print(nn_model.summary())`).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 23 (Блок 12).
* **Параметры:**
  * Входная размерность: `input_shape=(11,)` (задана в первом скрытом слое Dense).
  * Скрытые слои: 3 слоя с 64, 32 и 16 нейронами.
* **Вывод `model.summary()`:**
  * Всего параметров: **$3\,921$ весов** (из них обучаемых: $3\,793$, необучаемых параметров BatchNormalization: $128$).
  * Слой `dense_1`: $11 \times 64 + 64 = 768$ параметров.
  * Слой `dense_2`: $64 \times 32 + 32 = 2\,080$ параметров.
  * Слой `dense_3`: $32 \times 16 + 16 = 528$ параметров.
  * Выходной `output`: $16 \times 1 + 1 = 17$ параметров.

---

### 30. Где задаются функции активации? Объясните выбор числа нейронов и активации выходного слоя для вашей задачи.
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L274-L280):** строки 274, 277, 279 (`relu`), строка 280 (`sigmoid`).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 23 (Блок 12).
* **Обоснование:**
  * Скрытые слои: `relu` ($f(z) = \max(0, z)$) обеспечивает быстрое вычисление градиентов без затухания.
  * Выходной слой: содержит **ровно 1 нейрон** с активацией **`sigmoid`** $\sigma(z) = \frac{1}{1 + e^{-z}}$, так как задача является бинарной. Сигмоида возвращает число из отрезка $[0, 1]$, интерпретируемое как апостериорная вероятность $P(Exited=1 | \mathbf{x})$.

---

### 31. Покажите вызов compile. Как выбраны функция потерь, оптимизатор и метрики с учётом формата целевой переменной?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L283-L287):** строки 283–287.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 23 (Блок 12).
* **Фрагмент кода:**
  ```python
  nn_model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      loss='binary_crossentropy',
      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
  )
  ```
* **Обоснование выбора:**
  * `loss='binary_crossentropy'`: стандартная функция логистических потерь (Log-Loss) для скалярного бинарного таргета $y \in \{0, 1\}$.
  * `optimizer=Adam`: оптимизатор с адаптивным шагом и экспоненциальным сглаживанием моментов (скорость $lr=0.001$).
  * `metrics=['accuracy', AUC(name='auc')]`: мониторинг общей доли верных ответов и площади под ROC-кривой.

---

### 32. Где задаются learning_rate, batch_size и количество эпох? Покажите, как изменить каждый из этих гиперпараметров.
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L284):** строка 284 (`learning_rate`), строки 297–298 (`epochs`, `batch_size`).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 23 (Блок 12).
* **Как изменить:**
  * `learning_rate`: внутри оптимизатора: `Adam(learning_rate=0.0005)` (уменьшение шага в 2 раза).
  * `batch_size`: в вызове `fit`: `model.fit(..., batch_size=64)` (увеличение пакета до 64).
  * `epochs`: в вызове `fit`: `model.fit(..., epochs=50)` (увеличение предельного числа эпох).

---

### 33. Покажите вызов fit (нейронная сеть). Какие данные используются для обучения и валидации? Где сохраняется история обучения?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L294-L301):** строки 294–301.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 23 (Блок 12).
* **Фрагмент кода:**
  ```python
  history = nn_model.fit(
      X_train_scaled, y_train,
      validation_split=0.2,
      epochs=40,
      batch_size=32,
      callbacks=[tb_callback, early_stop],
      verbose=1
  )
  ```
* **Данные:** 80% от `X_train_scaled` ($6\,400$ строк) идут на вычисление градиентов и обновление весов, а 20% (`validation_split=0.2`, $1\,600$ строк) выделяются для валидации в конце каждой эпохи.
* **История обучения:** возвращается в объекте `history` и доступна через словарь `history.history` (`loss`, `val_loss`, `accuracy`, `val_accuracy`, `auc`, `val_auc`).

---

### 34. Где нейронная сеть формирует предсказания на тестовых данных? Как её выход преобразуется в метки классов и используется для расчёта метрик?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L304-L306):** строки 304–306.
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 23 (Блок 12).
* **Фрагмент кода:**
  ```python
  y_prob_nn = nn_model.predict(X_test_scaled, verbose=0).ravel()
  y_pred_nn = (y_prob_nn >= 0.5).astype(int)
  evaluate_model('Neural Network (TensorFlow)', y_test, y_pred_nn, y_prob_nn)
  ```
* **Преобразование:** непрерывные вероятности $P \in [0, 1]$ бинаризуются по стандартному решающему порогу $0.5$. Полученный дискретный массив `y_pred_nn` передается в функции расчета `accuracy_score`, `precision_score`, `recall_score`, `f1_score`.

---

### 35. Покажите, как организовано сравнение нескольких конфигураций нейронной сети. Где сохраняются параметры и результаты запусков, позволяющие оценить влияние изменяемого гиперпараметра?
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L267-L270):** строки 267–270 (создание директорий логов), строка 289, строки 311–327 (сводная таблица).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 23 (Блок 12).
* **Где сохраняются результаты:**
  * При каждом запуске в каталоге `logs/fit/` создается именованная папка с таймстампом (`ann_baseline_...`, `ann_experiment_...`).
  * В TensorBoard на вкладке **Scalars** кривые разных запусков автоматически сопоставляются на единой координатной сетке, позволяя сравнивать сходимость различных функций активации или оптимизаторов.

---

### 36. Покажите создание callback TensorBoard, передачу его в fit и настройку каталога логов. Запустите TensorBoard, откройте два эксперимента и объясните различия между кривыми обучения и валидации.
* **Навигация по коду:**
  * 📄 **В скрипте [`lab2_code.py`](file:///D:/Stud/AI/Lab2/defense/01_code/lab2_code.py#L267-L270):** строки 267–270 (путь к логам), строки 289–291 (создание коллбэков), строка 299 (передача в `fit`).
  * 📓 **В блокноте [`Lab2.ipynb`](file:///D:/Stud/AI/Lab2/Lab2.ipynb):** Ячейка 23 (Блок 12).
* **Фрагмент кода:**
  ```python
  log_base_dir = os.path.join('logs', 'fit')
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir_nn = os.path.join(log_base_dir, f"ann_baseline_{timestamp}")
  os.makedirs(log_dir_nn, exist_ok=True)

  tb_callback = TensorBoard(log_dir=log_dir_nn, histogram_freq=1)
  early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

  history = nn_model.fit(..., callbacks=[tb_callback, early_stop])
  ```
* **Команда запуска TensorBoard в терминале:**
  ```bash
  tensorboard --logdir logs/fit
  ```
* **Объяснение кривых обучения и валидации:**
  * **Train Loss vs Val Loss:** функция потерь на обучении монотонно снижается с $0.48$ до $0.34$. На валидации ошибка стабилизируется в районе 23–25 эпохи на уровне $0.3288$. Коллбэк `EarlyStopping` фиксирует оптимальные веса (`restore_best_weights=True`) и предотвращает расхождение кривых (переобучение).
