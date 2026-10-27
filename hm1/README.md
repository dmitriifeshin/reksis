# ДЗ 1 по рекомендательным системам

**Студент:** Фешин Дмитрий Витальевич

**Датасет:** Amazon Reviews 2023, категория Video Games (McAuley Lab, UCSD)

- страница датасета: https://amazon-reviews-2023.github.io/
- файлы: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
  - оценки: `benchmark/5core/rating_only/Video_Games.csv`
  - метаданные: `raw/meta_categories/meta_Video_Games.jsonl`

Данные в репозиторий не загружены, ноутбук сам скачивает их в `data/` при первом запуске.

## Структура

```
experiments.ipynb      основной ноутбук: данные, EDA, протокол, подбор, сравнение на test, выводы
models/
  item_knn.py          item-based kNN (косинус с шринкингом)
  ials.py              iALS, решение через сопряжённые градиенты
  ease.py              EASE
  fm.py                факторизационная машина с метапризнаками, BPR (PyTorch)
  ials_custom.py       iALS с кастомным таргетом (оценка × затухание по времени)
  popular.py           TopPopular, бейзлайн
utils/
  data.py              загрузка, k-core, временное разбиение, ground truth
  features.py          признаки товаров из метаданных
  metrics.py           топ-K и метрики (NDCG, Recall, HitRate, Coverage)
```

## Запуск

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 experiments.ipynb
```
