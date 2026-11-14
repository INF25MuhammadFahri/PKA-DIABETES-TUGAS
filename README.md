# 🩺 Diabetes KNN — Multi Preprocessing Comparison

Proyek ini membandingkan akurasi model **KNN** dengan berbagai metode *imputation* (preprocessing missing values) pada dataset diabetes.

---

## 📁 Struktur Proyek

```
PKA-DIABETES-TUGAS/
├── diabetes.csv                  ← Dataset utama (raw)
├── main.py                       ← Script utama
├── requirements.txt              ← Daftar dependency
└── output/                       ← Hasil generate (auto-created)
    ├── copy1_raw.csv
    ├── copy2_knn_imputed.csv
    ├── copy3_mean_imputed.csv
    ├── copy4_median_imputed.csv
    ├── copy5_median_per_class.csv
    └── accuracy_comparison.csv
```

---

## ⚙️ Instalasi

```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan

```bash
python main.py
```

---

## 🔧 Konfigurasi Global (`main.py`)

| Variable | Default | Keterangan |
|----------|---------|------------|
| `K_MANUAL` | `5` | Nilai K untuk KNN manual |
| `TEST_SIZE` | `0.2` | Rasio data test (20%) |
| `RANDOM_STATE` | `42` | Seed reproducibility |

---

## 📊 Alur Eksperimen

```
diabetes.csv (raw)
      │
      ├─ Copy 1 : Raw (tanpa preprocessing)        → KNN + KNN GridSearch
      ├─ Copy 2 : KNN Imputation                   → KNN + KNN GridSearch
      ├─ Copy 3 : Mean Imputation (Global)         → KNN + KNN GridSearch
      ├─ Copy 4 : Median Imputation (Global)       → KNN + KNN GridSearch
      └─ Copy 5 : Median Per Class (per Outcome)   → KNN + KNN GridSearch
                                                            │
                                              Perbandingan Akurasi
                                           (accuracy_comparison.csv)
```

**Missing values** ditangani pada kolom: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
(nilai 0 pada kolom-kolom tersebut tidak mungkin secara biologis → dianggap missing)

---

## 📈 Output

- **5 file CSV** (satu per copy/metode preprocessing)
- **`accuracy_comparison.csv`** — tabel perbandingan akurasi semua metode
- Tabel perbandingan di terminal
