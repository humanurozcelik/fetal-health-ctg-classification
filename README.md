# CTG Verileriyle Fetal Sağlık Sınıflandırması (Fetal Health Classification)

Bu proje, kardiyotokografi (CTG) cihazlarından elde edilen dijital sinyal özelliklerini kullanarak anne karnındaki fetüsün sağlık durumunu (Normal, Şüpheli, Patolojik) tahmin eden bir makine öğrenmesi iş akışıdır.

Projenin detaylı metodolojik anlatımına, klinik çıkarımlarına ve süreç adımlarına aşağıdaki Medium makalesinden ulaşabilirsiniz:

https://medium.com/@humanurozcelik555/ctg-verileriyle-fetal-sa%C4%9Fl%C4%B1k-s%C4%B1n%C4%B1fland%C4%B1rmas%C4%B1-%C3%BC%C3%A7-makine-%C3%B6%C4%9Frenmesi-modelinin-kar%C5%9F%C4%B1la%C5%9Ft%C4%B1r%C4%B1lmas%C4%B1-c5b962e5379a?sharedUserId=humanurozcelik555

## Projenin Amacı ve Yaklaşımı
Tıbbi verilerde sıklıkla karşılaşılan **sınıf dengesizliği (class imbalance)** problemiyle başa çıkmak ve makine öğrenmesi algoritmalarının fetal sağlık sınıflandırmasındaki performansını değerlendirmek amaçlanmıştır. 

Çalışmada metodolojik doğruluğu sağlamak adına:
* Sınıf dengesizliğini ele almak için `class_weight='balanced'` parametresi kullanılmıştır.
* Veri sızıntısını (Data Leakage) önlemek amacıyla ölçekleme işlemleri **Scikit-Learn Pipeline** içerisine entegre edilmiştir.
* Model seçimi, test setine hiç dokunulmadan sadece eğitim seti üzerinde **5-Fold Cross Validation** ve **Macro F1** skoru ile yapılmıştır.

## Veri Seti
Kullanılan veri seti, 2126 gözlemden oluşan açık kaynaklı "Fetal Health Classification" (CTG) veri setidir. 
* **Özellikler (Features):** Kalp atım hızı, variabilite, hızlanmalar ve yavaşlamalar dahil olmak üzere 21 klinik parametre.
* **Hedef Değişken (Target):** `fetal_health` (1.0: Normal, 2.0: Şüpheli, 3.0: Patolojik)

*Not: Veri temizleme aşamasında 13 adet tekrarlanan (duplicate) veri setinden çıkarılmış ve model 2.113 benzersiz vaka üzerinden eğitilmiştir.*

## Kullanılan Teknolojiler ve Modeller
* **Dil:** Python
* **Kütüphaneler:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, xlrd
* **Karşılaştırılan Modeller:** Lojistik Regresyon, Destek Vektör Makineleri (SVM), Random Forest (Rastgele Orman)
* **Optimizasyon:** GridSearchCV

##  Kurulum ve Çalıştırma

Projeyi kendi ortamınızda çalıştırmak için:

1. Repository'yi klonlayın

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xlrd
```

3. Veri setinin `fetal_health.xls` adıyla ana dizinde olduğundan emin olun ve kodu çalıştırın:
```bash
python fetal-health-ctg-classification.py
```

## Sonuçlar ve Değerlendirme
Çapraz doğrulama sonuçlarına göre en yüksek performansı **Random Forest** algoritması (0.885 CV Macro F1) göstermiştir. 

Grid Search ile optimize edilen Random Forest modelinin daha önce hiç görmediği **Hold-out Test Seti (%20)** üzerindeki nihai performansı:
* **Accuracy (Doğruluk):** %94.6
* **Macro F1 Skoru:** 0.91
* **Patolojik Sınıf Recall (Duyarlılık):** %97 (35 patolojik vakanın 34'ü doğru sınıflandırılmıştır).

Random Forest Feature Importance analizine göre model, abnormal_short_term_variability ve percentage_of_time_with_abnormal_long_term_variability özelliklerine en yüksek göreli önem değerlerini atamıştır. Bu değerler, modelin sınıflandırma kararlarında bu özelliklerden daha fazla yararlandığını gösterir; klinik nedensellik anlamına gelmez.

---
*Bu çalışma bir eğitim/makine öğrenmesi uygulamasıdır ve doğrudan klinik tanı aracı olarak kullanılmak üzere doğrulanmamıştır.*
