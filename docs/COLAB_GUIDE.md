# Running on Google Colab

This guide covers running SR-FedOpt on Google Colab.
(یک راهنمای فارسی نیز پیوست شده است / A Persian version follows.)

---

## Visualization Troubleshooting

If you see `No .npz results found under results/` when running the visualizer:

```bash
# Make sure you are in the project root first:
import os
os.chdir('/content/drive/MyDrive/sr-fedopt')
!python tools/visualize_results.py
```

Or use the wrapper script which auto-detects the project root:
```bash
!python visualize.py
```

---

# اجرا روی Google Colab

این راهنما برای اجرای پروژه SR-FedOpt روی Google Colab است.

## گام‌های اجرا

### ۱. آپلود پروژه به Google Drive

1. کل پوشه `sr-fedopt` را به Google Drive خود آپلود کنید
2. می‌توانید از Desktop app یا رابط وب Drive استفاده کنید
3. مسیر پیشنهادی: `MyDrive/sr-fedopt/`

### ۲. باز کردن نوت‌بوک Colab

**روش الف) مستقیم از Drive:**
1. به Google Drive بروید
2. فایل `run_on_colab.ipynb` را پیدا کنید
3. کلیک راست → Open with → Google Colaboratory

**روش ب) آپلود دستی:**
1. به [Google Colab](https://colab.research.google.com) بروید
2. File → Upload notebook
3. فایل `run_on_colab.ipynb` را آپلود کنید

### ۳. فعال‌سازی GPU

**⚠️ مهم:** حتماً GPU را فعال کنید:
- Runtime → Change runtime type
- Hardware accelerator: **GPU** (T4 یا بهتر)
- Save

### ۴. تنظیم مسیر پروژه

در سلول دوم نوت‌بوک، مسیر را تغییر دهید:

```python
# مسیر پروژه در Google Drive - این را تغییر دهید!
PROJECT_PATH = '/content/drive/MyDrive/sr-fedopt'
```

### ۵. اجرای سلول‌ها

سلول‌ها را **به ترتیب** اجرا کنید (Shift+Enter):

1. ✅ Mount Drive
2. ✅ تنظیم مسیر
3. ✅ نصب وابستگی‌ها
4. ✅ بررسی GPU
5. ✅ اجرای آزمایش‌ها

## استراتژی‌های اجرا

### الف) تست سریع (۵ دقیقه)
برای بررسی اولیه که همه‌چیز کار می‌کند:
```python
!python federated_learning.py --schedule test
```

### ب) اجرای کامل (۶-۸ ساعت)
اجرای همه ۱۸۰ آزمایش:
```python
!python federated_learning.py --schedule main
```

### ج) اجرای تدریجی (پیشنهادی ⭐)
برای جلوگیری از timeout، آزمایش‌ها را در batch‌های کوچک اجرا کنید:

```python
# Session 1 (۲ ساعت)
!python federated_learning.py --schedule main --start 0 --end 30

# Session 2 (۲ ساعت)
!python federated_learning.py --schedule main --start 30 --end 60

# Session 3 (۲ ساعت)
!python federated_learning.py --schedule main --start 60 --end 90

# ...و ادامه
```

**مزیت:** اگر Colab قطع شد، از جایی که رسیده‌اید ادامه می‌دهید!

## مانیتورینگ اجرا

### بررسی پیشرفت:
```python
# تعداد آزمایش‌های تکمیل‌شده
!find results/main -name '*.npz' | wc -l
```

### مشاهده لاگ‌ها:
```python
# آخرین نتایج
!tail -20 results/main/mnist/sigma0p0/fedavg/run42/*.npz
```

## نکات مهم ⚠️

### ۱. محدودیت زمانی Colab
- **رایگان:** حدود ۱۲ ساعت session
- **Colab Pro:** ۲۴ ساعت
- **راه حل:** اجرای تدریجی با `--start` و `--end`

### ۲. قطع شدن session
اگر Colab قطع شد:
1. دوباره Mount کنید
2. به پوشه پروژه بروید
3. از `--start X` برای ادامه استفاده کنید

### ۳. فضای دیسک
- نتایج در Drive شما ذخیره می‌شوند
- ۱۸۰ آزمایش ≈ ۲۰۰-۳۰۰ MB

### ۴. مشکلات رایج

**خطای "CUDA out of memory":**
```python
# batch_size را کاهش دهید
"batch_size": [1]  # قبلاً تنظیم شده
```

**خطای "Module not found":**
```python
# دوباره نصب کنید
!pip install -r requirements.txt
```

**خطای مسیر:**
```python
# مسیر را چک کنید
import os
print(os.getcwd())
print(os.listdir('.'))
```

## تولید نمودارها

بعد از اتمام آزمایش‌ها:

```python
# تولید همه نمودارها
!python tools/visualize_results.py

# نمایش در نوت‌بوک
from IPython.display import Image
Image(filename='results/plots/mnist_sigma0p0_all_methods_combined.png')
```

## دانلود نتایج

### روش ۱: دانلود مستقیم از Colab
```python
!zip -r results.zip results/
from google.colab import files
files.download('results.zip')
```

### روش ۲: از Drive (پیشنهادی)
نتایج از قبل در Drive شما هستند! فقط:
1. به Drive بروید
2. پوشه `sr-fedopt/results/` را دانلود کنید

## تخمین زمان اجرا

| Dataset | Model | Experiments | زمان تقریبی |
|---------|-------|-------------|-------------|
| MNIST | logistic | 60 | ~۲ ساعت |
| FashionMNIST | logistic | 60 | ~۲ ساعت |
| CIFAR10 | simple_cnn | 60 | ~۳-۴ ساعت |
| **جمع** | | **۱۸۰** | **~۶-۸ ساعت** |

## پشتیبانی

اگر مشکلی داشتید:
1. خروجی خطا را بررسی کنید
2. GPU را چک کنید: `torch.cuda.is_available()`
3. مسیرها را بررسی کنید
4. نوت‌بوک را Restart کنید

---

**نکته:** همیشه نتایج را از Drive backup بگیرید! 🎯
