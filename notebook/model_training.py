import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import os

# ---------------------------
# 1. Download dataset automatically if missing
# ---------------------------
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_PATH = "titanic.csv"

if not os.path.exists(DATA_PATH):
    print("📥 Downloading Titanic dataset...")
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)
else:
    print("✅ Using existing Titanic dataset...")
    df = pd.read_csv(DATA_PATH)

# Fix any potential column case issues
if 'Survived' not in df.columns and 'survived' in df.columns:
    df.rename(columns={'survived': 'Survived'}, inplace=True)

# ---------------------------
# 2. Select useful columns
# ---------------------------
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
df = df[cols].copy()

# Features & target
X = df.drop('Survived', axis=1)
y = df['Survived']

# ---------------------------
# 3. Preprocessing pipelines
# ---------------------------
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ---------------------------
# 4. Create full pipeline with Logistic Regression
# ---------------------------
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# ---------------------------
# 5. Split data & train model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)

# ---------------------------
# 6. Save the trained model
# ---------------------------
joblib.dump(clf, "model.pkl")
print("✅ Model trained and saved as model.pkl")

# ---------------------------
# 7. Print test accuracy
# ---------------------------
acc = clf.score(X_test, y_test)
print(f"📊 Test Accuracy: {acc:.4f}")
