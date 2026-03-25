# sLLM 70M Training Pipeline

Полный `single-node` пайплайн для обучения малой авторегрессионной модели порядка `~70M` параметров:

- обучение BPE-токенизатора;
- подготовка pretraining-данных в бинарные `uint16`-шарды;
- предобучение causal LM;
- annealing на длинном контексте `8192`;
- SFT на диалоговом датасете;
- базовая генерация и оценка perplexity.

## Что внутри

- `src/sllm/config.py` - dataclass-конфиги и JSON helper-функции.
- `src/sllm/model.py` - Llama-подобный декодер: `RoPE + RMSNorm + SwiGLU + tied embeddings`.
- `src/sllm/data.py` - shard writer, random pretrain dataset, fixed-length SFT dataset.
- `src/sllm/checkpoint.py` - сохранение и загрузка checkpoint.
- `scripts/train_tokenizer.py` - обучение BPE.
- `scripts/prepare_pretrain_data.py` - токенизация и запись pretraining-шардов.
- `scripts/train_pretrain.py` - основное предобучение.
- `scripts/prepare_sft_data.py` - упаковка диалогового датасета в фиксированные тензоры.
- `scripts/train_sft.py` - supervised fine-tuning.
- `scripts/eval_perplexity.py` - оценка валидaционной perplexity.
- `scripts/generate.py` - простая генерация текста.
- `configs/*.json` - стартовые конфиги под `RTX 5090`.
- `LOGGING.md` - описание текстовых логов и JSONL-метрик.

## Архитектура

Базовая конфигурация лежит в `configs/model_70m.json`:

- `32` слоя
- `d_model = 384`
- `6` голов внимания (`head_dim = 64`)
- `ffn_hidden_dim = 1024`
- `vocab_size = 49152`
- `max_seq_len = 8192`
- `bias-free` линейные слои внутри трансформера
- `weight tying` между входным эмбеддингом и LM head

Такая конфигурация практичнее для бюджета `~70M`, чем `32 x 512`, который на простой MHA-реализации уже уходит далеко за целевой размер.

## Зависимости

Нужна Python-среда с пакетами:

- `torch`
- `datasets`
- `tokenizers`
- `numpy`

Опционально полезны:

- `transformers` для дальнейшей интеграции с HF-экосистемой;
- современный `PyTorch` с CUDA, где `scaled_dot_product_attention` уже умеет использовать быстрые attention kernels.

## Hugging Face Hub (QED-75M)

Экспорт чекпоинта в формат Hub (веса, токенизатор, `config.json`, **remote code** `modeling_qed.py`):

```bash
python scripts/export_to_huggingface.py
# с загрузкой на Hub: задайте HUGGING_FACE_HUB_TOKEN и добавьте --push
```

Исходник кода для Hub лежит в `hf_hub/modeling_qed.py` (классы `QEDConfig`, `QEDForCausalLM`).

У пользователя загрузка выглядит так (нужен `trust_remote_code=True` для модели):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "levossadtchi/QED-75M",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("levossadtchi/QED-75M")
```

## Важные замечания по датасетам

- Стартовый микс задан в `configs/data_mix_10b.json`.
- Идентификаторы Hugging Face датасетов и названия их полей иногда меняются. Если какой-то источник на вашей стороне называется иначе, правьте только JSON-конфиг, а не код.
- Для `The Stack` может потребоваться доступ к датасету и/или выбор конкретного языка.
- Если вы хотите другой состав смеси, меняйте только веса и dataset ids в `configs/data_mix_10b.json`.
- На этапе `prepare_pretrain_data.py` источники теперь не записываются последовательно "целиком по датасету". Вместо этого используется глобальный interleaving по весам смеси, чтобы train-поток был равномернее перемешан по доменам.

## Порядок запуска

### 1. Обучить токенизатор

```bash
python scripts/train_tokenizer.py \
  --data-config configs/data_mix_10b.json \
  --output-dir data/tokenizer \
  --vocab-size 49152
```

Результат:

- `data/tokenizer/tokenizer.json`
- `data/tokenizer/tokenizer_meta.json`

### 2. Подготовить pretraining-данные

```bash
python scripts/prepare_pretrain_data.py \
  --data-config configs/data_mix_10b.json \
  --tokenizer-dir data/tokenizer \
  --output-dir data/pretokenized
```

Результат:

- `data/pretokenized/train/*.bin`
- `data/pretokenized/val/*.bin`
- `data/pretokenized/dataset_summary.json`

Как это работает:

- train и val собираются как глобальные interleaved потоки;
- следующий источник выбирается по недовыполнению своей целевой доли;
- итоговая смесь держится близко к заданным весам уже в самом pretokenized корпусе;
- один и тот же документ не режется одновременно в `val` и `train`.
- токенизация документов идет пакетно через `encode_batch`, а не по одному документу, чтобы ускорить подготовку корпуса.

### 3. Dry run предобучения

Сначала лучше сделать короткую проверку на нескольких десятках шагов.

```bash
python scripts/train_pretrain.py \
  --model-config configs/model_70m.json \
  --train-config configs/pretrain_5090_stage1.json \
  --max-steps 50
```

Что смотреть:

- `loss` должен уверенно снижаться;
- не должно быть `nan`/`inf`;
- checkpoints должны сохраняться в `checkpoints/pretrain_stage1`.

Если loss неадекватен, сначала уменьшайте `learning_rate` в `configs/pretrain_5090_stage1.json` до `1e-3`.

### 3.1 Dry run на Mac / MPS

Если вы хотите просто проверить, что пайплайн запускается на `MacBook` через `MPS`, используйте отдельный облегченный конфиг:

```bash
python scripts/train_pretrain.py \
  --model-config configs/model_70m.json \
  --train-config configs/pretrain_mps_dryrun.json
```

Этот режим не предназначен для реального обучения модели. Он нужен только для локальной sanity-check проверки загрузки данных, forward/backward и checkpointing.

### 4. Основной stage 1 pretraining

```bash
python scripts/train_pretrain.py \
  --model-config configs/model_70m.json \
  --train-config configs/pretrain_5090_stage1.json
```

Это основной проход на `seq_len=2048`.

### 5. Stage 2 annealing на длинном контексте

```bash
python scripts/train_pretrain.py \
  --model-config configs/model_70m.json \
  --train-config configs/pretrain_5090_stage2_anneal.json
```

Этот этап поднимает фактическое обучение к `8192` токенам и продолжает модель из `checkpoints/pretrain_stage1/last.pt`.

### 6. Подготовить SFT-данные

```bash
python scripts/prepare_sft_data.py \
  --config configs/sft_data_smoltalk.json \
  --tokenizer-dir data/tokenizer \
  --output-dir data/sft/processed \
  --seq-len 2048
```

Результат:

- `data/sft/processed/train_input_ids.bin`
- `data/sft/processed/train_labels.bin`
- `data/sft/processed/train_metadata.json`
- `data/sft/processed/val_metadata.json`

Стартовый `SFT`-микс в `configs/sft_data_smoltalk.json` использует несколько subset'ов `HuggingFaceTB/smoltalk`, а не один общий `all`:

- `smol-magpie-ultra` (`40%`) как основной general instruct/reasoning источник;
- `openhermes-100k` (`15%`) для более широкой instruct-вариативности;
- `self-oss-instruct` (`15%`) для сохранения навыков по коду;
- `everyday-conversations` (`1%`) как небольшая примесь естественного multi-turn диалога;
- `numina-cot-100k` (`10%`) для reasoning;
- `metamathqa-50k` (`5%`) как умеренная math-добавка;
- `longalign` (`1.5%`) для небольшой примеси длинных запросов и ответов;
- `HuggingFaceH4/ultrachat_200k` (`12.5%`, `train_sft`) как крупный внешний general-chat/instruct источник для добора объема.

Для `smol-magpie-ultra` дополнительно включен фильтр `row_filters: {"quality": "good"}`. То есть в основной общий источник `SFT` попадают только примеры, помеченные как `good`.

Вес `everyday-conversations` и `longalign` намеренно уменьшен: эти subset'ы полезны по стилю данных, но слишком малы, чтобы на них выделять крупную долю из `200k` train-примеров. Основной объем теперь безопасно добирается через `ultrachat_200k`.

Сознательно не включены `smol-constraints`, `smol-rewrite`, `smol-summarize`, `explore-instruct-rewriting`, `apigen-80k` и `systemchats-30k`, потому что они слишком узко задают поведение модели и могут заметно сместить базовый чатовый `SFT`.

`prepare_sft_data.py` теперь умеет читать смесь `sources[]` с весами, применять per-source фильтры по полям датасета, отдельно собирать `train/val` по каждому источнику и сохранять per-source summary в `dataset_summary.json`.

При упаковке длинных диалогов `prepare_sft_data.py` теперь старается сохранить supervision:

- сначала пробует обычное окно от начала примера;
- если в этом окне слишком мало assistant-токенов, берет хвост последовательности;
- если после упаковки supervised токенов все равно меньше, чем `min_supervised_tokens`, пример пропускается.

В стартовом конфиге используется `min_supervised_tokens = 16`, чтобы не записывать практически пустые `SFT`-примеры без полезного обучающего сигнала.

Важно: `SFT`-labels формируются как next-token targets только для assistant-частей диалога. То есть модель обучается предсказывать следующий assistant-токен, а не копировать текущий токен на той же позиции. После изменения этой логики готовый `SFT`-датасет нужно пересобрать заново.

### 7. Запустить SFT

```bash
python scripts/train_sft.py \
  --model-config configs/model_70m.json \
  --train-config configs/sft_5090.json
```

Инициализация по умолчанию идет из `checkpoints/pretrain_stage2/last.pt`.

### 8. Оценить perplexity

```bash
python scripts/eval_perplexity.py \
  --checkpoint checkpoints/pretrain_stage2/last.pt \
  --data-dir data/pretokenized \
  --seq-len 2048 \
  --batch-size 8 \
  --batches 50
```

### 9. Проверить генерацию

```bash
python scripts/generate.py \
  --checkpoint checkpoints/sft/last.pt \
  --tokenizer-dir data/tokenizer \
  --prompt "Объясни, что такое градиентный спуск простыми словами." \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-k 50
```

## Resume

Для продолжения обучения не нужно менять код:

- укажите `resume_from` в `configs/pretrain_5090_stage1.json`;
- или укажите `resume_from` в `configs/sft_5090.json`.

Пример:

```json
{
  "resume_from": "checkpoints/pretrain_stage1/last.pt"
}
```

## Практические советы под RTX 5090

- Начинайте с `micro_batch_size`, уже указанного в конфиге. Если память заканчивается, уменьшайте сначала `micro_batch_size`, а не `seq_len`.
- Если памяти остается много, повышайте `micro_batch_size` и уменьшайте `grad_accum_steps`, сохраняя нужный глобальный batch.
- Для первой отладки не запускайте сразу `10B` токенов. Сначала проверьте весь pipeline на сильно уменьшенной смеси данных.
- `seq_len=8192` лучше использовать как отдельную annealing-стадию, а не с первого шага.

## Что проверять после каждого этапа

- После токенизатора: есть `tokenizer.json`, словарь близок к `49152`.
- После pretokenization: появились `.bin`-шарды и `dataset_summary.json`.
- После dry run: loss падает, checkpoint сохраняется.
- После stage 1/2: `eval` loss стабильно идет вниз.
- После SFT: модель начинает отвечать в формате диалога, а не только продолжать сырой текст.

## Ограничения текущей реализации

- Это практичный `single-node` пайплайн, а не распределенное обучение.
- В генерации нет KV-cache; для отладки и коротких ответов этого достаточно.
- SFT-упаковка сейчас делает фиксированную длину с pad/truncate. Для production-цикла позже можно добавить packing нескольких коротких диалогов в один пример.
- В коде используется `torch.nn.functional.scaled_dot_product_attention`, а не жесткая зависимость на внешний `flash-attn`.
