# 🎓 CLS Model Training Guide

Panduan lengkap untuk melatih classifier models (CLS) untuk prediksi arah trading.

## 📋 Isi

1. [Persiapan](#persiapan)
2. [Cara Training Model](#cara-training-model)
3. [Penjelasan Model](#penjelasan-model)
4. [Evaluasi Model](#evaluasi-model)
5. [Troubleshooting](#troubleshooting)

---

## 🔧 Persiapan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Konfigurasi MT5

Edit file `.env`:

```env
# MT5 Credentials - DEMO (untuk training)
MT5_LOGIN_DEMO=12345678
MT5_PASSWORD_DEMO=YourPassword
MT5_SERVER_DEMO=MetaQuotes-Demo
```

### 3. Pastikan MT5 Terminal Running

- Buka MetaTrader 5
- Login ke akun DEMO
- Pastikan koneksi aktif

---

## 🚀 Cara Training Model

### Metode 1: Training Otomatis (Semua Timeframes)

```bash
python models/cls_trainer.py
```

Script ini akan:
- ✅ Mengumpulkan 5000 candles untuk setiap timeframe (M5, M15, H1, H4)
- ✅ Menambahkan technical indicators
- ✅ Membuat features engineering
- ✅ Melatih model Random Forest
- ✅ Menyimpan model (.pkl) dan scaler ke `models/saved_models/`

**Durasi:** 10-30 menit tergantung spesifikasi komputer

### Metode 2: Training Manual (Python Script)

```python
import logging
from core.mt5_handler import MT5Handler
from models.cls_trainer import CLSModelTrainer
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Connect MT5
mt5 = MT5Handler(
    login=int(os.getenv('MT5_LOGIN_DEMO')),
    password=os.getenv('MT5_PASSWORD_DEMO'),
    server=os.getenv('MT5_SERVER_DEMO')
)

if mt5.initialize():
    # Create trainer
    trainer = CLSModelTrainer(output_dir="./models/saved_models")
    
    # Train all timeframes
    trainer.train_all_timeframes(
        mt5_handler=mt5,
        symbol='XAUUSD',
        model_type='random_forest'  # atau 'gradient_boosting'
    )
    
    mt5.shutdown()
```

### Metode 3: Training Single Timeframe

Jika hanya ingin retrain satu timeframe:

```python
# ... (koneksi MT5 sama seperti di atas)

trainer.retrain_single_timeframe(
    mt5_handler=mt5,
    timeframe='M5',  # M5, M15, H1, atau H4
    symbol='XAUUSD',
    model_type='random_forest'
)
```

---

## 📊 Penjelasan Model

### Model Architecture

Bot menggunakan **Random Forest Classifier** dengan konfigurasi:

```python
RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=15,            # Maximum tree depth
    min_samples_split=10,    # Minimum samples to split
    min_samples_leaf=5,      # Minimum samples per leaf
    class_weight='balanced', # Handle imbalanced data
    n_jobs=-1               # Use all CPU cores
)
```

### Features (Input)

Model menggunakan **50+ features** termasuk:

#### 1. Price Action Features
- Body percentage (candle body size)
- Range percentage (high-low range)
- Close position (where price closed in range)
- Upper/lower wicks

#### 2. Technical Indicators
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands** (position and width)
- **Stochastic** (K and D lines)
- **EMA** (9, 21, 50, 200)
- **ATR** (Average True Range)

#### 3. Trend Features
- EMA alignment (bullish/bearish)
- Trend direction
- Distance from moving averages

#### 4. Volume Features
- Volume ratio vs average
- Volume momentum

#### 5. Momentum Features
- ROC (Rate of Change)
- Momentum indicators
- Recent price changes

#### 6. Time Features
- Hour of day
- Day of week
- Trading session (London, NY, Asian)

### Labels (Output)

Model memprediksi 3 kelas:

- **0 = SELL** (Harga akan turun)
- **1 = HOLD** (Tidak ada signal jelas)
- **2 = BUY** (Harga akan naik)

Label dibuat dengan **Adaptive Labeling Method**:
- Menggunakan ATR untuk threshold dinamis
- Melihat 10 candles ke depan
- Membandingkan upside vs downside potential

---

## 📈 Evaluasi Model

### Metrics yang Ditampilkan

Setelah training, script menampilkan:

#### 1. Accuracy Scores
```
Train accuracy: 85.2%
Test accuracy: 72.5%
CV accuracy: 71.8% (+/- 3.2%)
```

**Interpretasi:**
- **Train accuracy** harus tinggi (>80%) tapi tidak terlalu tinggi (overfitting)
- **Test accuracy** menunjukkan performa real-world
- **CV accuracy** lebih reliable dari test accuracy
- Gap antara train dan test tidak boleh >15% (overfitting)

#### 2. Classification Report

```
              precision    recall  f1-score   support

        SELL       0.65      0.70      0.67       250
        HOLD       0.80      0.75      0.77       300
         BUY       0.68      0.72      0.70       245

    accuracy                           0.72       795
```

**Interpretasi:**
- **Precision**: Dari yang diprediksi BUY, berapa % yang benar
- **Recall**: Dari semua BUY yang sebenarnya, berapa % yang tertangkap
- **F1-score**: Harmonic mean precision & recall (metric terbaik)

**Target yang Baik:**
- F1-score > 0.65 untuk BUY dan SELL
- Accuracy > 70%

#### 3. Confusion Matrix

```
[[175  50  25]     # SELL: 175 benar, 50 salah ke HOLD, 25 ke BUY
 [ 45 225  30]     # HOLD: 225 benar
 [ 30  40 175]]    # BUY: 175 benar
```

**Interpretasi:**
- Diagonal (175, 225, 175) = prediksi benar
- Off-diagonal = prediksi salah
- Idealnya diagonal tinggi, off-diagonal rendah

#### 4. Feature Importance

```
Top 20 Important Features:
  rsi: 0.0850
  macd_normalized: 0.0720
  ema_bullish_alignment: 0.0650
  ...
```

**Interpretasi:**
- Shows which features the model relies on most
- RSI and MACD typically very important
- Helps understand model's decision-making

---

## ✅ Model Quality Checklist

Model **SIAP DIGUNAKAN** jika:

- ✅ Test accuracy > 70%
- ✅ CV accuracy > 68%
- ✅ F1-score (BUY) > 0.65
- ✅ F1-score (SELL) > 0.65
- ✅ Gap train-test accuracy < 15%
- ✅ Tidak ada error saat training

Model **PERLU DIPERBAIKI** jika:

- ❌ Test accuracy < 65%
- ❌ F1-score < 0.60
- ❌ Train accuracy >> Test accuracy (overfitting)
- ❌ Label distribution sangat tidak seimbang

---

## 🔄 Retraining Schedule

### Kapan Harus Retrain?

1. **Rutin** (Recommended)
   - Setiap 1 bulan untuk M5, M15
   - Setiap 2 bulan untuk H1, H4

2. **Performance-Based**
   - Jika win rate turun < 50%
   - Jika model confidence sering rendah
   - Setelah perubahan market regime (contoh: Fed policy change)

3. **Data-Based**
   - Setelah mengumpulkan 1000+ candles baru
   - Setelah event market major (NFP, FOMC, dll)

### Cara Cepat Retrain

```bash
# Retrain semua
python models/cls_trainer.py

# Atau retrain satu timeframe
python -c "
from models.cls_trainer import CLSModelTrainer
from core.mt5_handler import MT5Handler
mt5 = MT5Handler(login, password, server)
mt5.initialize()
trainer = CLSModelTrainer()
trainer.retrain_single_timeframe(mt5, 'M5')
mt5.shutdown()
"
```

---

## 🛠️ Troubleshooting

### Problem: "No CLS models found"

**Solusi:**
1. Pastikan sudah running `python models/cls_trainer.py`
2. Check folder `models/saved_models/` ada file:
   - `cls_m5.pkl`, `cls_m15.pkl`, `cls_h1.pkl`, `cls_h4.pkl`
   - `scaler_m5.pkl`, `scaler_m15.pkl`, dll

### Problem: "Not enough data"

**Solusi:**
1. Pastikan MT5 terminal running
2. Check symbol XAUUSD tersedia
3. Download historical data di MT5:
   - Tools → History Center
   - Double click XAUUSD
   - Download data

### Problem: Low accuracy (<65%)

**Penyebab:**
- Data tidak cukup
- Label threshold tidak optimal
- Features tidak representative

**Solusi:**
1. Collect lebih banyak data (10000+ candles)
2. Adjust labeling threshold di `data/preprocessor.py`:
   ```python
   threshold=0.20  # Increase dari 0.15
   ```
3. Try gradient_boosting model:
   ```python
   model_type='gradient_boosting'
   ```

### Problem: Overfitting (train 95%, test 65%)

**Solusi:**
1. Reduce model complexity:
   ```python
   max_depth=10  # Dari 15
   min_samples_split=20  # Dari 10
   ```
2. Add more training data
3. Use cross-validation

### Problem: Memory Error

**Solusi:**
1. Reduce candles:
   ```python
   candles=3000  # Dari 5000
   ```
2. Train one timeframe at a time
3. Close other applications

---

## 📁 Output Files

Setelah training, file berikut akan dibuat:

```
models/saved_models/
├── cls_m5.pkl          # Model M5
├── scaler_m5.pkl       # Feature scaler M5
├── cls_m15.pkl         # Model M15
├── scaler_m15.pkl      # Feature scaler M15
├── cls_h1.pkl          # Model H1
├── scaler_h1.pkl       # Feature scaler H1
├── cls_h4.pkl          # Model H4
└── scaler_h4.pkl       # Feature scaler H4
```

**PENTING:** Jangan delete file scaler! Model tidak akan work tanpa scaler yang matching.

---

## 💡 Tips untuk Model Terbaik

1. **Use Quality Data**
   - Gunakan akun dengan historical data lengkap
   - Avoid data dari weekend (spread tinggi)

2. **Balanced Labels**
   - Check label distribution saat training
   - Idealnya: 30-40% BUY, 20-30% HOLD, 30-40% SELL

3. **Feature Engineering**
   - Model sudah include 50+ features
   - Bisa tambah features custom di `data/preprocessor.py`

4. **Ensemble Strategy**
   - Bot sudah combine CLS dengan strategy signals
   - Trend Fusion menggunakan weighted voting

5. **Regular Monitoring**
   - Track model accuracy via logs
   - Compare predicted vs actual results
   - Retrain jika performance drop

---

## 📞 Support

Jika ada masalah saat training:

1. Check logs di console output
2. Pastikan semua dependencies terinstall
3. Verify MT5 connection
4. Check available disk space (need ~100MB)

---

**⚡ Selamat Training! ⚡**

*Remember: Model quality = Data quality + Feature quality + Training quality*

