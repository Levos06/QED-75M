# LOGGING

Этот документ описывает, какие лог-файлы создает проект, где они лежат, какие поля в них пишутся и как использовать JSONL-метрики для визуализации.

## Общая идея

В проекте есть два уровня логирования:

1. обычные текстовые `.log` файлы для чтения человеком;
2. структурированные `.jsonl` файлы для машинной обработки, графиков и аналитики.

Текстовые логи подходят для:

- просмотра прогресса в реальном времени;
- поиска ошибок;
- отладки загрузки данных, checkpoint и конфигов.

JSONL подходит для:

- построения графиков `loss`, `lr`, `tok/s`;
- пост-анализа обучения;
- импорта в pandas, DuckDB, ClickHouse, notebook или любой dashboard.

## Где лежат лог-файлы

### Токенизатор

Папка:

- `data/tokenizer/logs/`

Файлы:

- `train_tokenizer_<timestamp>.log`

### Подготовка pretraining-данных

Папка:

- `data/pretokenized/logs/`

Файлы:

- `prepare_pretrain_data_<timestamp>.log`

### Подготовка SFT-данных

Папка:

- `data/sft/processed/logs/`

Файлы:

- `prepare_sft_data_<timestamp>.log`

### Pretraining

Папка:

- `outputs/pretrain_stage1/logs/`
- `outputs/pretrain_stage2/logs/`

Файлы:

- `train_pretrain_<timestamp>.log`
- `train_pretrain_<timestamp>.jsonl`

### SFT

Папка:

- `outputs/sft/logs/`

Файлы:

- `train_sft_<timestamp>.log`
- `train_sft_<timestamp>.jsonl`

### Eval

Папка:

- `outputs/eval/logs/`

Файлы:

- `eval_perplexity_<timestamp>.log`

### Generation

Папка:

- `outputs/generate/logs/`

Файлы:

- `generate_<timestamp>.log`

## Формат имен

Во всех логах используется timestamp формата:

- `YYYYMMDD_HHMMSS`

Пример:

- `train_pretrain_20260311_181530.log`
- `train_pretrain_20260311_181530.jsonl`

Это означает, что один запуск создает собственную пару логов и не перезаписывает предыдущие.

## Что пишется в обычные `.log`

Обычные логи создаются через `logging` и содержат строки вида:

```text
2026-03-11 18:15:30,123 | INFO | Train step | step=100 loss=3.4211 lr=0.001234 ...
```

В них обычно попадают:

- старт и завершение запуска;
- аргументы CLI;
- эффективные конфиги;
- пути к логам;
- загрузка checkpoint;
- summary по устройству и датасету;
- шаги обучения;
- eval;
- сохранения checkpoint;
- прогресс подготовки данных;
- итоговая summary по обработанным данным.

## Что пишется в `.jsonl`

JSONL создается только для этапов обучения:

- `train_pretrain.py`
- `train_sft.py`

Каждая строка в `.jsonl` это один JSON-объект.

Пример:

```json
{"event":"train","timestamp":"2026-03-11T18:15:30","step":100,"loss":3.4211,"lr":0.001234}
```

Такой формат удобен тем, что:

- его можно читать построчно;
- файл не нужно целиком загружать в память;
- он хорошо совместим с pandas и shell-инструментами.

## События в JSONL

В `.jsonl` используются следующие `event`.

### `run_started`

Пишется в самом начале запуска.

Поля:

- `event`
- `timestamp`
- `log_path`
- `metrics_path`
- `model_config`
- `train_config`
- `args`

Назначение:

- зафиксировать полную конфигурацию конкретного запуска;
- упростить воспроизводимость;
- связать `.log` и `.jsonl`.

### `runtime_summary`

Пишется после инициализации модели, датасета и runtime.

Для pretraining содержит:

- `device`
- `precision`
- `compile_model`
- `parameters`
- `seq_len`
- `micro_batch_size`
- `grad_accum_steps`
- `tokens_per_step`
- `num_train_shards`
- `train_dir`
- `val_dir`

Для SFT содержит:

- `device`
- `precision`
- `compile_model`
- `parameters`
- `seq_len`
- `micro_batch_size`
- `grad_accum_steps`
- `tokens_per_step`
- `dataset_path`
- `train_examples`
- `val_examples`

Назначение:

- быстро понять runtime-режим;
- не искать эти параметры по текстовому логу.

### `resumed`

Пишется, если обучение продолжается из `resume_from`.

Поля:

- `event`
- `timestamp`
- `step`
- `checkpoint`

Назначение:

- зафиксировать точку resume;
- понимать, что запуск не был начат с нуля.

### `initialized_from_checkpoint`

Сейчас пишется в `train_sft.py`, если модель инициализируется checkpoint'ом, но optimizer state не восстанавливается как resume.

Поля:

- `event`
- `timestamp`
- `checkpoint`

Назначение:

- отличать fine-tuning from scratch от fine-tuning из pretrain checkpoint.

### `train`

Это основное событие. Пишется каждые `log_interval` шагов.

Поля:

- `event`
- `timestamp`
- `step`
- `loss`
- `lr`
- `tok_per_sec`
- `grad_norm`
- `tokens_seen`
- `elapsed_sec`
- `seq_len`
- `micro_batch_size`
- `grad_accum_steps`

Если используется CUDA, дополнительно пишутся:

- `allocated_gb`
- `reserved_gb`
- `max_allocated_gb`
- `max_reserved_gb`

Назначение:

- строить графики обучения;
- смотреть throughput;
- отслеживать расход VRAM;
- выявлять нестабильность по `grad_norm`.

### `eval`

Пишется каждые `eval_interval` шагов.

Поля:

- `event`
- `timestamp`
- `step`
- `val_loss`
- `perplexity`
- `eval_batches`

Назначение:

- строить график validation loss;
- определять, есть ли улучшение;
- смотреть момент деградации или переобучения.

### `checkpoint`

Пишется каждый раз при сохранении checkpoint.

Поля:

- `event`
- `timestamp`
- `step`
- `step_checkpoint`
- `last_checkpoint`
- `tokens_seen`

Назначение:

- понимать, какие checkpoint соответствуют каким шагам;
- удобно поднимать обучение или анализировать историю запусков.

### `run_finished`

Пишется в конце запуска.

Поля:

- `event`
- `timestamp`
- `final_step`
- `tokens_seen`

Назначение:

- видеть корректное завершение run;
- легко отделять полный run от оборванного.

## Поля текстовых логов по этапам

### `train_tokenizer.py`

Пишет:

- старт запуска;
- путь к `.log`;
- аргументы запуска;
- summary tokenizer config;
- старт каждого источника;
- итоговый размер словаря;
- id спецтокенов;
- пути сохранения tokenizer и metadata.

### `prepare_pretrain_data.py`

Пишет:

- старт запуска;
- путь к `.log`;
- tokenizer metadata;
- старт каждого источника;
- target train/val токены по каждому источнику;
- progress каждые `10_000` документов;
- число записанных шардов;
- итоговую summary по всем источникам.

### `prepare_sft_data.py`

Пишет:

- старт запуска;
- путь к `.log`;
- конфиг источника SFT;
- progress каждые `5_000` примеров;
- число train/val примеров;
- итоговые metadata.

### `train_pretrain.py`

Пишет:

- старт запуска;
- путь к `.log` и `.jsonl`;
- model/train config;
- runtime summary;
- resume/init checkpoint;
- шаги обучения;
- eval;
- checkpoint events;
- завершение.

### `train_sft.py`

Пишет:

- старт запуска;
- путь к `.log` и `.jsonl`;
- model/SFT config;
- runtime summary;
- resume/init checkpoint;
- шаги обучения;
- eval;
- checkpoint events;
- завершение.

### `eval_perplexity.py`

Пишет:

- старт запуска;
- путь к `.log`;
- аргументы;
- итоговые `val_loss` и `perplexity`.

### `generate.py`

Пишет:

- старт запуска;
- путь к `.log`;
- аргументы;
- число токенов prompt и выхода.

## Как читать JSONL в pandas

Пример:

```python
import pandas as pd

df = pd.read_json("outputs/pretrain_stage1/logs/train_pretrain_20260311_181530.jsonl", lines=True)
train_df = df[df["event"] == "train"].copy()
eval_df = df[df["event"] == "eval"].copy()
```

График loss:

```python
train_df.plot(x="step", y="loss")
```

График validation loss:

```python
eval_df.plot(x="step", y="val_loss")
```

График throughput:

```python
train_df.plot(x="step", y="tok_per_sec")
```

## Как быстро посмотреть JSONL без Python

Например, последние записи:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/pretrain_stage1/logs/train_pretrain_20260311_181530.jsonl")
lines = path.read_text(encoding="utf-8").splitlines()
for line in lines[-5:]:
    print(json.loads(line))
PY
```

Или фильтрация только `eval` событий:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/pretrain_stage1/logs/train_pretrain_20260311_181530.jsonl")
for line in path.read_text(encoding="utf-8").splitlines():
    obj = json.loads(line)
    if obj["event"] == "eval":
        print(obj)
PY
```

## На что смотреть в первую очередь

Для pretraining:

- `loss`
- `val_loss`
- `perplexity`
- `lr`
- `tok_per_sec`
- `grad_norm`
- `max_reserved_gb`

Для SFT:

- `loss`
- `val_loss`
- `perplexity`
- `grad_norm`
- `tok_per_sec`

Для подготовки данных:

- число обработанных документов;
- train/val token targets;
- число shard-файлов;
- итоговую summary.

## Как интерпретировать некоторые поля

### `loss`

Это средний training loss за последний интервал логирования, а не loss одного-единственного micro-step.

### `tok_per_sec`

Это эффективная скорость в токенах на optimizer steps, уже учитывающая `grad_accum_steps`.

### `grad_norm`

Это значение, полученное перед optimizer step после `unscale`, если используется `fp16`, и после clipping.

Очень полезно для отладки:

- внезапный скачок может указывать на нестабильность;
- `nan` или бесконечность почти всегда сигнализируют о проблеме.

### `allocated_gb` и `reserved_gb`

- `allocated_gb` показывает реально используемую память;
- `reserved_gb` показывает память, удерживаемую CUDA allocator.

Обычно `reserved_gb` выше, чем `allocated_gb`, и это нормально.

### `max_allocated_gb` и `max_reserved_gb`

Это пиковые значения за весь запуск на данный момент.

Именно на них удобнее всего ориентироваться при подборе `micro_batch_size`.

## Практические сценарии использования

### Отладка dry run

Смотрите:

- есть ли событие `run_started`;
- идут ли `train` события;
- сохраняется ли `checkpoint`;
- заканчивается ли run событием `run_finished`.

### Поиск регрессии по скорости

Сравнивайте:

- `tok_per_sec`
- `allocated_gb`
- `reserved_gb`

между разными запусками.

### Подбор batch size

Смотрите:

- `max_reserved_gb`
- `max_allocated_gb`

Если запас большой, можно увеличить `micro_batch_size`.

### Поиск проблем с качеством

Смотрите:

- уменьшается ли `train loss`;
- уменьшается ли `val_loss`;
- не взрывается ли `grad_norm`;
- не перестает ли улучшаться `perplexity`.

## Ограничения текущей системы логирования

- JSONL сейчас создается только для этапов обучения, а не для подготовки данных.
- Логи не ротируются автоматически.
- Нет встроенного TensorBoard/W&B backend.
- Нет автоматического агрегатора по нескольким run.

## Что можно добавить позже

- JSONL и для data-prep стадий;
- экспорт графиков в PNG после завершения run;
- TensorBoard writer;
- интеграцию с Weights & Biases;
- отдельный `metrics_summary.json` с последними агрегированными значениями;
- системные метрики CPU/RAM/диска на каждый интервал.
