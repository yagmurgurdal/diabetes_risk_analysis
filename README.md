# Diyabet Riski Tahmini Projesi

Bu proje `diabetes.csv` veri seti kullanilarak hazirlandi ve **10 farkli siniflandirma algoritmasi** ayni veri akisi uzerinde egitilip karsilastirildi.

## Kullanilan 10 Algoritma

- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- Gaussian Naive Bayes
- MLP

## Projede Yapilanlar

- Veri seti okunur.
- Tibben anlamsiz `0` degerleri eksik veri gibi ele alinir.
- Median ile doldurma uygulanir.
- Ek risk odakli ozellikler uretilir.
- Veri egitim, dogrulama ve test olarak ayrilir.
- 10 algoritma ayni veri yapisi ile egitilir.
- Her model icin threshold secilir.
- Accuracy, Precision, Recall, F1-Score ve ROC-AUC hesaplanir.
- En iyi model otomatik secilir.

## Calistirma

```powershell
C:\Users\yagmu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe diabetes_project.py
```

## Uretilen Dosyalar

- `results/model_comparison.csv`
- `results/best_model_predictions.csv`
- `results/metrics.json`
- `results/project_summary.md`

## Su Anki En Iyi Model

Son calistirmada en iyi sonuc:
- Model: `Logistic Regression`
- Accuracy: `0.7532`
- Precision: `0.6053`
- Recall: `0.8519`
- F1-Score: `0.7077`
- ROC-AUC: `0.8161`

Bu sonuc, dogrudan test verisi uzerinde 10 algoritmanin ayni kosullarda karsilastirilmasiyla elde edilmistir.
