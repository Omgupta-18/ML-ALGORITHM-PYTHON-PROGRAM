import numpy as np
import pandas as pd

df1 = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)

dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')

X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.6,
    random_state=0,
    stratify=y
)

# Use 100,000 test samples
if len(X_test) > 100000:
    X_test, _, y_test, _ = train_test_split(
        X_test,
        y_test,
        train_size=100000,
        random_state=0,
        stratify=y_test
    )

# Use 50,000 stratified training samples
if len(X_train) > 50000:
    X_train, _, y_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=50000,
        random_state=0,
        stratify=y_train
    )

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Kernel Approximation
from sklearn.kernel_approximation import Nystroem

kpca = Nystroem(
    kernel='rbf',
    gamma=15,
    n_components=10,
    random_state=0
)

X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)


from sklearn.svm import SVC

classifier = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=False,
    cache_size=2000,
    random_state=0
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n")
print(cm)

print("\nAccuracy : {:.4f}".format(
    accuracy_score(y_test, y_pred)
))

print("\nPrecision : {:.4f}".format(
    precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
))

print("\nRecall : {:.4f}".format(
    recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
))

print("\nF1 Score : {:.4f}".format(
    f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
))

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))