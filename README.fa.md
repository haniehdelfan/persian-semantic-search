# جست‌وجوی معنایی فارسی: مقایسه‌ی مدل‌های Embedding

[English](README.md) | **فارسی**

هدف این پروژه نشان‌دادن کار عملی با متن فارسی و **مقایسه‌ی مدل‌ها با داده** است، نه صرفاً انتخاب بر اساس شهرت. خروجی نهایی یک جدول «کیفیت در برابر سرعت و حجم» و پاسخ مستند به این پرسش است: *برای متن فارسی کدام مدل را انتخاب می‌کنم و چرا؟*

## معماری

```
 ویکی‌پدیای فارسی ──► prepare_data ──► data/corpus.jsonl  (چند هزار سند: ابتدای مقاله‌ها)
                                              │
 build_eval (Claude + بازبینی دستی) ──► data/queries.jsonl  (۱۰۰ پرسش، هرکدام یک سند درست)
                                              │
        ┌─────────────────────────────────────┴───────────────┐
        ▼                                                     ▼
   run_eval                                            build_index  ──►  artifacts/index/<model>/
   4 مدل Embedding + BM25                                                         │
   × نرمال‌سازی (none / full / هر مرحله جدا)                                      ▼
   × پرسش تمیز / پرسش «نویزی»                                         FastAPI: POST /search
        │
        ▼
   results/ (json, csv, md, png)  ──►  بلوک نتایج همین README
```

## ساختار

```
src/pss/
  normalize.py     نرمال‌سازی فارسی؛ هر مرحله یک flag مستقل
  perturb.py       شبیه‌سازی ورودی واقعی کاربر (کیبورد عربی، ارقام عربی، بدون نیم‌فاصله)
  models.py        رجیستری مدل‌ها + prefix مخصوص E5
  metrics.py       MRR، recall@k، nDCG، بازه‌ی اطمینان bootstrap
  bm25.py          baseline لغوی
  bench.py         اجرای بنچمارک (کیفیت، تأخیر، حجم)
  report.py        جدول markdown، نمودار، پیشنهاد خودکار
  prepare_data.py  ساخت corpus
  build_eval.py    ساخت مجموعه‌ی ارزیابی
  run_eval.py      اجرای کامل مقایسه
  build_index.py   ساخت ایندکس برای یک مدل
  api.py           سرویس FastAPI
tests/             ۱۹ تست (نرمال‌سازی، متریک‌ها، pipeline و API بدون دانلود مدل)
```

## اجرا

```bash
pip install -e ".[ml,dev]"      # یا: pip install -r requirements.txt و export PYTHONPATH=src

make data         # نمونه‌ی تصادفی ۵۰۰۰ مقاله از ویکی‌پدیای فارسی (HF: wikimedia/wikipedia, 20231101.fa)
make eval-set     # ۱۰۰ پرسش با Claude (نیاز به ANTHROPIC_API_KEY)  → بازبینی: خط پرسش‌های مبهم را از data/queries.jsonl حذف کنید (queries_review.csv فقط برای خواندن است)
make eval         # مقایسه‌ی کامل؛ بلوک نتایج همین README خودکار به‌روز می‌شود
make index MODEL=e5-base
make serve MODEL=e5-base
make test
```

پیش از اجرای مدل‌های واقعی می‌توانید کل pipeline را بدون دانلود چیزی امتحان کنید: `make smoke`.

### اجرا روی GitHub Actions (بدون نصب روی کامپیوتر)

کل پروژه روی runnerهای CPU رایگان گیتهاب اجرا می‌شود (ریپوی public: ۴ هسته و ۱۶ گیگ رم؛ هر job حداکثر ۶ ساعت):

1. secret با نام `ANTHROPIC_API_KEY` را اضافه کنید (Settings ← Secrets and variables ← Actions).
2. تب Actions ← **Build evaluation set** ← Run workflow. پرسش‌های پیش‌نویس در `data/` commit می‌شوند.
3. در مرورگر بازبینی کنید: `data/queries_review.csv` را (به‌صورت جدول) بخوانید و خط پرسش‌های مبهم را از `data/queries.jsonl` پاک کنید.
4. تب Actions ← **Run benchmark** ← Run workflow. نتایج در `results/` و بلوک نتایج README commit می‌شود.

تأخیرها روی یک runner مشترک اندازه‌گیری می‌شوند؛ آن‌ها را فقط برای مقایسه‌ی نسبی مدل‌ها بخوانید، نه عدد مطلق production.

## طراحی ارزیابی

**داده.** هر سند ابتدای یک مقاله است (حداکثر ۱۲۰۰ نویسه) تا در حد ۵۱۲ توکن مدل‌ها بگنجد. مقاله‌ها با reservoir sampling انتخاب می‌شوند تا نمونه تصادفی باشد. فهرست‌ها و صفحه‌های ابهام‌زدایی کنار گذاشته می‌شوند. به‌جای ویکی‌پدیا می‌توان corpus خبری خودتان را داد: `--input-jsonl`.

**پرسش‌ها.** برای هر سند نمونه‌شده، Claude یک پرسش طبیعی می‌نویسد که پاسخش در همان متن است، بدون کپی عنوان و بدون جمله‌ی عیناً برداشته‌شده. سند مبدأ همان «سند درست» است. خروجی یک پیش‌نویس است: پرسش‌های مبهم (که چند مقاله می‌توانند پاسخشان باشند) باید در مرحله‌ی بازبینی حذف یا اصلاح شوند. حالت `--mode title` هم برای تست آفلاین هست، ولی به دلیل شباهت لغوی عنوان و متن فقط یک sanity check است.

**متریک‌ها.** چون هر پرسش یک سند درست دارد: `MRR@10` (میانگین معکوس رتبه‌ی سند درست، تا رتبه‌ی ۱۰)، `recall@1/5/10` و `nDCG@10`. برای MRR بازه‌ی اطمینان ۹۵٪ با bootstrap گزارش می‌شود؛ با ۱۰۰ پرسش بازه‌ها عریض‌اند و اختلاف‌های کوچک‌تر از آن معنادار نیستند.

**سرعت و حجم.** تأخیر یک پرسش (batch=۱، بعد از warm-up) به‌صورت p50 و p95 برای encode و جداگانه برای جست‌وجو؛ توان encode سند (سند بر ثانیه)؛ تعداد پارامتر، وزن fp32 و حجم ایندکس. دستگاه (CPU/GPU) در `results.csv` ثبت می‌شود و اعداد فقط روی همان دستگاه قابل مقایسه‌اند.

**مدل‌ها.** `multilingual-e5-small`، `multilingual-e5-base`، `paraphrase-multilingual-MiniLM-L12-v2`، `bge-m3` و BM25 به‌عنوان baseline. مدل‌های E5 با prefixهای `query: ` و `passage: ` اجرا می‌شوند. طول ورودی به ۵۱۲ توکن محدود شده، به‌جز MiniLM که همان پیش‌فرض خودش (۱۲۸ توکن) را دارد؛ این یعنی روی اسناد بلند فقط ابتدای متن را «می‌بیند» و در نتیجه باید با همین قید خوانده شود. مدل‌های دیگر (`e5-large`, `labse`) در `models.py` هستند و با `--models` اضافه می‌شوند.

## نرمال‌سازی

| مرحله | کار |
|---|---|
| `chars` | NFKC و یکسان‌سازی ي→ی، ك→ک، ة→ه، ۀ→ه، أ/إ→ا |
| `diacritics` | حذف اعراب و کشیده |
| `digits` | ارقام فارسی و عربی → ASCII |
| `punct` | ؟ ، ؛ « » → معادل ASCII |
| `zwnj_fix` | استفاده‌ی استاندارد از نیم‌فاصله: «می شود»→«می‌شود»، «کتاب ها»→«کتاب‌ها»، حذف نیم‌فاصله‌ی تکراری یا بی‌جا |
| `zwnj_space` | جایگزینی هر نیم‌فاصله با فاصله (برای مقایسه با `zwnj_fix`) |
| `whitespace` | فشرده‌کردن فاصله‌ها |

**اثر نرمال‌سازی جدا اندازه‌گیری می‌شود** (هر مرحله تنها، در برابر `none`) و روی دو مجموعه پرسش: `clean` و `noisy`. ویکی‌پدیا از قبل تمیز است، پس انتظار می‌رود روی پرسش تمیز اثر نرمال‌سازی نزدیک صفر باشد؛ مجموعه‌ی `noisy` همان پرسش‌ها را با ي/ك، ارقام عربی و بدون نیم‌فاصله می‌نویسد تا اثر آن روی ورودی واقعی کاربر دیده شود. هزینه‌ی ablation کم است چون embedding متن‌های بدون تغییر دوباره محاسبه نمی‌شود.

محدودیت شناخته‌شده: بازگرداندن نیم‌فاصله فقط با قاعده‌های امن (پیشوند می/نمی، پسوند ها/های/تر/ترین) انجام می‌شود. نیم‌فاصله‌ی درون واژه‌های مرکب مانند «جهان‌گستر» اگر کاربر آن را ننویسد بازیابی نمی‌شود (در تست‌ها مستند شده).

## نتایج

<!-- RESULTS:START -->
*هنوز اجرا نشده. بعد از `make data && make eval-set && make eval` این بلوک خودکار با جدول مقایسه، اثر نرمال‌سازی، ablation و پیشنهاد مدل پر می‌شود.*
<!-- RESULTS:END -->

نمودار کیفیت در برابر تأخیر: `results/tradeoff.png`.

## نتیجه‌گیری

> این بخش را بعد از اجرای `make eval` با اعداد خودتان بنویسید (۳ تا ۵ خط).

قاعده‌ی انتخاب که گزارش خودکار هم از آن استفاده می‌کند: **کوچک‌ترین مدلی که MRR@10 آن حداکثر ۰٫۰۲ از بهترین مدل کمتر باشد.** قالب پیشنهادی:

- برای متن فارسی **«X»** را انتخاب می‌کنم، چون MRR@10 آن …، recall@5 آن … و p50 تأخیر … میلی‌ثانیه است.
- «Y» بهترین کیفیت را دارد، ولی … برابر کندتر و … برابر بزرگ‌تر است و این اختلاف در بازه‌ی اطمینان … (معنادار / نامعنادار) است.
- نرمال‌سازی روی پرسش تمیز … و روی پرسش نویزی … اثر دارد؛ مرحله‌ی مؤثرتر … بود.
- BM25 در برابر بهترین مدل … (اگر BM25 نزدیک است، یعنی پرسش‌ها لغوی‌اند و باید پرسش‌های دشوارتر اضافه کرد).

## API

```bash
PSS_INDEX_DIR=artifacts/index/e5-base uvicorn pss.api:app --port 8000

curl -X POST localhost:8000/search -H 'content-type: application/json' \
     -d '{"query": "پایتخت ایران کدام شهر است؟", "k": 5}'
```

`GET /search?q=...&k=5`، `GET /info` (مدل و تنظیمات نرمال‌سازی ایندکس) و `GET /health` هم هست. پرسش پیش از encode با **همان** نرمال‌سازی‌ای که ایندکس با آن ساخته شده پردازش می‌شود (در `meta.json` ذخیره است) و در پاسخ `normalized_query` و `took_ms` برمی‌گردد. جست‌وجو brute-force با ضرب ماتریسی است؛ برای چند هزار سند کافی است و برای میلیون‌ها سند باید به FAISS یا مشابه رفت.

## محدودیت‌ها

- پرسش‌ها مصنوعی‌اند (LLM-generated)، پس سبک آن‌ها با پرسش واقعی کاربر فرق دارد؛ بازبینی دستی این خطا را کم می‌کند ولی از بین نمی‌برد.
- یک سند درست برای هر پرسش؛ اگر سند دیگری هم پاسخ را داشته باشد، به مدل جریمه می‌خورد (برای همه‌ی مدل‌ها یکسان).
- فقط ابتدای مقاله‌ها ایندکس می‌شود، نه کل متن.
- تأخیرها به سخت‌افزار وابسته‌اند.

---

## English summary

Benchmarks 4 multilingual embedding models (multilingual-e5-small/base, paraphrase-multilingual-MiniLM-L12-v2, bge-m3) plus a BM25 baseline on a random sample of Persian Wikipedia lead paragraphs, using ~100 LLM-written questions (manually reviewed) with one gold document each. Reports MRR@10 (with bootstrap CIs), recall@1/5/10, nDCG@10, single-query latency, throughput, and model/index size. Persian normalization (letter/digit unification, diacritics, ZWNJ repair, punctuation) is ablated step by step on clean and on simulated-noisy queries (Arabic keyboard, Arabic-Indic digits, missing half-space). A FastAPI service serves any built index and applies the index's own normalization to queries. Selection rule: the smallest model within 0.02 MRR@10 of the best.
