"""
Fetal Sağlık Durumu Sınıflandırması - Medium Final Projesi (Nihai Versiyon)
Problem Türü: Çok Sınıflı Sınıflandırma (1: Normal, 2: Şüpheli, 3: Patolojik)
Metodoloji: Duplicate Temizliği, Veri Sızıntısı Koruması (Pipeline), 5-Fold CV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Veri Yükleme ve Temizleme
print("--- Adım 1: Veri Yükleme ve Temizleme ---")
# Veri seti excel (xls) formatında okunuyor
df = pd.read_excel("fetal_health.xls")

print(f"İlk Veri Seti Boyutu: {df.shape}")
print("Eksik değer sayısı:", df.isnull().sum().sum())
print("Tekrarlanan satır sayısı:", df.duplicated().sum())

# Tekrarlanan satırları çıkarma
df = df.drop_duplicates().reset_index(drop=True)
print(f"Temizlenmiş Veri Seti Boyutu: {df.shape}")

# Grafik 1: Sınıf Dağılımı Çubuk Grafiği
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='fetal_health', hue='fetal_health', palette='Set2', legend=False)
plt.title('Fetal Sağlık Sınıf Dağılımı (1: Normal, 2: Şüpheli, 3: Patolojik)')
plt.ylabel('Vaka Sayısı')
plt.xlabel('Sağlık Durumu')
plt.tight_layout()
plt.savefig('1_class_distribution.png', dpi=300)

# Özellikler Arası Korelasyon Analizi
plt.figure(figsize=(14, 12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Klinik Parametreler Arası Korelasyon Isı Haritası')
plt.tight_layout()
plt.savefig('2_correlation_heatmap.png', dpi=300)

# Veri Bölme (Test Setini Korumaya Alma)
X = df.drop("fetal_health", axis=1)
y = df["fetal_health"]

# Test setine (Hold-out) sadece en sonda bakılacak! Veri Sızıntısını (Data Leakage) önlemek için önce ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

#Modellerin Çapraz Doğrulama (CV) ile Karşılaştırılması
print("\n--- Adım 4: 5-Fold Cross Validation ile Model Karşılaştırması ---")

# Ölçekleme (StandardScaler) işlemini Pipeline içine alarak CV sırasında sızıntıyı engelliyoruz
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(class_weight='balanced', random_state=42))
])

# Random Forest ağaç tabanlı olduğu için ölçeklendirmeye ihtiyaç duymaz
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

models = {
    "Lojistik Regresyon": pipeline_lr,
    "SVM": pipeline_svm,
    "Random Forest": rf_model
}

cv_results = {}
for name, model in models.items():
    # Sınıf dengesizliği olduğu için Macro F1 skoru baz alınıyor
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    cv_results[name] = scores.mean()
    print(f"{name} 5-Fold CV Macro F1 Skoru: {scores.mean():.4f}")

# Grafik 5: Model Karşılaştırma Grafiği
plt.figure(figsize=(8, 5))
sns.barplot(x=list(cv_results.keys()), y=list(cv_results.values()), hue=list(cv_results.keys()), palette='viridis', legend=False)
plt.title('Modellerin 5-Fold Çapraz Doğrulama Performansları (Macro F1)')
plt.ylabel('Macro F1 Skoru')
plt.ylim(0, 1)
for i, v in enumerate(cv_results.values()):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('5_model_comparison.png', dpi=300)

# En İyi Model İçin Hiperparametre Optimizasyonu

print("\n--- Adım 5: Random Forest Optimizasyonu (Grid Search) ---")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# Optimizasyon sırasında da Macro F1 kullanıyoruz
grid_search = GridSearchCV(
    estimator=rf_model, 
    param_grid=param_grid, 
    cv=5, 
    scoring='f1_macro', 
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi CV Macro F1:", grid_search.best_score_)

# Nihai Test (Hold-out Test Seti)
print("\n--- Adım 6: Nihai Modelin Test Seti Performansı ---")
# Seçilen ve optimize edilen modelin daha önce hiç görmediği verideki performansı
y_pred_best = best_rf.predict(X_test)

print("\nOptimize Edilmiş Random Forest Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_best))

# Karışıklık Matrisi (Confusion Matrix)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Şüpheli', 'Patolojik'], 
            yticklabels=['Normal', 'Şüpheli', 'Patolojik'])

plt.title('Nihai Model Karışıklık Matrisi')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.tight_layout()
plt.savefig('3_confusion_matrix.png', dpi=300)

# Özellik Önem Dereceleri
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 8))
sns.barplot(x=importances[indices][:10], y=X.columns[indices][:10], hue=X.columns[indices][:10], palette='viridis', legend=False)
plt.title('Random Forest Modelinin En Fazla Önem Verdiği 10 CTG Özelliği')
plt.xlabel('Önem Derecesi')
plt.ylabel('Özellik (Feature)')
plt.tight_layout()
plt.savefig('4_feature_importance.png', dpi=300)

print("\nTüm metodolojik analizler tamamlandı ve görseller kaydedildi.")