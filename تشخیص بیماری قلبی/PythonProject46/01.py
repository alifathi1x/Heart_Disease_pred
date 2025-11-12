
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay)
import matplotlib.pyplot as plt
import seaborn as sns

# --------- تنظیمات ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --------- 1. بارگذاری دیتاست ----------
# فرض: فایل CSV به نام 'heart.csv' در همان پوشه قرار دارد.
# اگر ورژن شما نام ستون‌ها را متفاوت دارد، نام ستون‌ها را مطابق دیتاست اصلاح کنید.
df = pd.read_csv(r"C:\Users\Ali\PycharmProjects\PythonProject46\heart.csv")
print("Shape:", df.shape)
#print(df.head())

# --------- 2. بررسی سریع ----------
print("\n--- Info ---")
print(df.info())
print("\n--- Describe ---")
print(df.describe())

# بعضی mirror ها id یا origin دارند؛ اگر ستون id وجود دارد حذفش کن:
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# --------- 3. آماده‌سازی برچسب (binary) ----------
# برخی نسخه‌ها: 'num' در بازه 0-4؛ تبدیل کنیم به 0 یا 1
if 'num' in df.columns:
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])
elif 'target' not in df.columns and 'diagnosis' in df.columns:
    # در صورت نام‌گذاری دیگر
    df['target'] = (df['diagnosis'] > 0).astype(int)
    df = df.drop(columns=['diagnosis'])

print("\nTarget distribution:")
print(df['target'].value_counts())

# --------- 4. انتخاب ویژگی‌ها ----------
# معمولاً ویژگی‌های استاندارد (از UCI): age, sex, cp, trestbps, chol, fbs, restecg,
# thalach, exang, oldpeak, slope, ca, thal
# اگر فایل شما همه‌ی اینها نداره، از ستون‌های موجود استفاده کن.
features = [c for c in df.columns if c != 'target']
X = df[features]
y = df['target']

# --------- 5. بررسی مقادیر گمشده ----------
print("\nMissing values per column:")
print(X.isna().sum())

# اگر مقادیر NaN هستن، از ایمپیوت استفاده می‌کنیم (median برای عددی، most_frequent برای categoric)
# شناسایی ستون‌های عددی/کاتگوریکال:
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

print("\nNumeric cols:", num_cols)
print("Categorical cols:", cat_cols)

# --------- 6. تقسیم train/test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# --------- 7. Pipeline پیش‌پردازش ----------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='passthrough'  # برای هر ستون دیگه (در صورت وجود)
)

# --------- 8. مدل‌ها و grid search ----------
# Logistic Regression pipeline
pipe_lr = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

# Decision Tree pipeline (درخت به مقیاس‌بندی نیاز نداره ولی pipeline رو یکدست نگه می‌داریم)
pipe_dt = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))
])

# پارامترهای ساده برای جستجو
param_grid_lr = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs']
}

param_grid_dt = {
    'clf__max_depth': [3, 5, 7, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

gs_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=cv, scoring='roc_auc', n_jobs=-1)
gs_dt = GridSearchCV(pipe_dt, param_grid_dt, cv=cv, scoring='roc_auc', n_jobs=-1)

# --------- 9. آموزش مدل‌ها ----------
print("\nTraining Logistic Regression...")
gs_lr.fit(X_train, y_train)
print("Best LR params:", gs_lr.best_params_)
print("Best LR CV ROC AUC:", gs_lr.best_score_)

print("\nTraining Decision Tree...")
gs_dt.fit(X_train, y_train)
print("Best DT params:", gs_dt.best_params_)
print("Best DT CV ROC AUC:", gs_dt.best_score_)

# --------- 10. ارزیابی روی تست ----------
def evaluate_model(grid_search, X_test, y_test, model_name="model"):
    best = grid_search.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:,1] if hasattr(best, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    print(f"\n--- Evaluation: {model_name} ---")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("ROC AUC:", roc)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    return {'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1, 'roc':roc}

res_lr = evaluate_model(gs_lr, X_test, y_test, "Logistic Regression")
res_dt = evaluate_model(gs_dt, X_test, y_test, "Decision Tree")

# --------- 11. مقایسه و نمایش ROC curve (اگر موجود) ----------
plt.figure(figsize=(8,6))
if res_lr['roc'] is not None:
    RocCurveDisplay.from_estimator(gs_lr.best_estimator_, X_test, y_test, name='Logistic Regression')
if res_dt['roc'] is not None:
    RocCurveDisplay.from_estimator(gs_dt.best_estimator_, X_test, y_test, name='Decision Tree')
plt.title('ROC Curves')
plt.show()

# --------- 12. (اختیاری) نمایش درخت تصمیم اگر انتخاب شد ----------
best_dt = gs_dt.best_estimator_.named_steps['clf']
# نیاز به features پس از preprocessor: استخراج نام فیچرها ممکنه پیچیده باشه (onehot)
# برای نمایش ساده درخت با plot_tree:
plt.figure(figsize=(16,10))
plot_tree(best_dt, filled=True, feature_names=features, class_names=['NoHD','HD'], rounded=True, max_depth=3)
plt.title("Decision Tree (partial view)")
plt.show()

# --------- 13. ذخیره مدل (اختیاری) ----------
import joblib
joblib.dump(gs_lr.best_estimator_, 'best_logistic_pipeline.joblib')
joblib.dump(gs_dt.best_estimator_, 'best_tree_pipeline.joblib')
print("\nModels saved: best_logistic_pipeline.joblib , best_tree_pipeline.joblib")