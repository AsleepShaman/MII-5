import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Загрузка данных
cancer = pandas.read_csv('cancer.csv')
print(cancer)

# cancer.info()

# В целевом столбце "M or В" содержатся значения:
    # M - Malignant (злокачественное образование) 
    # B - Benign (доброкачественное образование)

cancer_array = cancer.values
x = cancer_array[:, 3:33]
# print(X)
y = cancer_array[:, 2]
# print(y)

# Разбиваем набор данных на 80% обучающих данных и 20% тестовых данных.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Используемые методы:
# 1. SGD Classifier (SGD) - Линейный классификатор с SGD-обучением
# 2. Random Forest Classifier (RF) - Случайный лес
# 3. Gaussian process classification (GP) - Гауссовская классификация

colors = ["#9370DB", "#00BFFF"]

classifiers = []
classifiers.append(('Линейный классификатор с SGD-обучением', SGDClassifier(max_iter = 1500, tol = 1e-4)))
classifiers.append(('Случайный лес', RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1)))
classifiers.append(('Гауссовская классификация', GaussianProcessClassifier()))

for name, clf in classifiers:
    print(name)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cv_results = accuracy_score(y_pred, y_test)
    print(classification_report(y_test, y_pred))
    
    # Графическое представление результатов     
    status = pandas.DataFrame(y_pred).value_counts()
    # print(status)
    fig, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.pie(status, labels=status.keys(), autopct='%1.1f%%', colors = colors)
    ax1.axis('equal')

status_true = pandas.DataFrame(y_test).value_counts()
# print(status_true)
fig, ax1 = plt.subplots()
ax1.set_title('Достоверные значения') 
ax1.pie(status_true, labels=status_true.keys(), autopct='%1.1f%%', colors = colors)
ax1.axis('equal')
plt.show()
