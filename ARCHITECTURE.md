# Архитектура проекта sLLM 70M

Этот документ подробно описывает внутреннее устройство проекта: общую схему пайплайна, архитектуру модели, форматы данных, ключевые проектные решения, классы, функции и связи между модулями.

## 1. Цель проекта

Проект реализует полный `single-node` pipeline для обучения небольшой авторегрессионной языковой модели порядка `~70M` параметров:

1. обучение BPE-токенизатора;
2. сбор и токенизация pretraining-корпуса;
3. запись токенов в компактный shard-формат для быстрого чтения;
4. предобучение causal language model;
5. annealing-стадия на длинном контексте;
6. supervised fine-tuning на диалоговых данных;
7. базовая оценка perplexity и генерация.

Проект намеренно не привязан к распределенному обучению, cluster orchestration и сложной инфраструктуре. Он ориентирован на практический запуск на одной современной GPU.

## 2. Общая схема пайплайна

Полный цикл выглядит так:

1. `configs/data_mix_10b.json` описывает смесь источников.
2. `scripts/train_tokenizer.py` обучает BPE-токенизатор на выборке из смеси.
3. `scripts/prepare_pretrain_data.py` загружает те же источники, interleave-смешивает их по весам, токенизирует и пишет в бинарные `uint16`-шарды.
4. `scripts/train_pretrain.py` читает эти шарды через `RandomTokenDataset` и обучает модель на задаче next-token prediction.
5. `scripts/train_pretrain.py` может использоваться повторно для stage 2 annealing с другим конфигом.
6. `scripts/prepare_sft_data.py` преобразует instruction/chat датасет в фиксированные обучающие примеры.
7. `scripts/train_sft.py` дообучает ту же модель на SFT-данных.
8. `scripts/eval_perplexity.py` проверяет качество на validation shards.
9. `scripts/generate.py` выполняет базовый inference.

## 3. Почему выбрана именно такая архитектура

### 3.1 Модель

Основной конфиг: `configs/model_70m.json`

- `n_layers = 32`
- `d_model = 384`
- `n_heads = 6`
- `head_dim = 64`
- `ffn_hidden_dim = 1024`
- `vocab_size = 49152`
- `max_seq_len = 8192`

### 3.2 Логика выбора

Ключевая идея: для очень маленькой модели важнее достаточная глубина, чем чрезмерная ширина.

Почему не `512 x 32`:

- при стандартном многоголовом attention без агрессивных оптимизаций модель слишком разрастается по параметрам;
- растут память и стоимость обучения;
- ухудшается практичность запуска на одной GPU.

Почему `384 x 32`:

- остается глубокая сеть с хорошей иерархией представлений;
- сохраняется бюджет около целевого масштаба `70M`;
- проще уложиться в VRAM и в throughput на consumer GPU.

### 3.3 Архитектурные решения

- Используется `RoPE`, потому что это современный и практичный способ кодирования позиций для длинного контекста.
- Используется `RMSNorm`, потому что он проще и дешевле `LayerNorm`, хорошо работает в pre-norm трансформерах.
- Используется `SwiGLU`, потому что он обычно дает более сильный MLP-блок, чем базовый GELU FFN.
- `bias=False` внутри основных линейных слоев повторяет практику современных decoder-only моделей.
- Используется `weight tying`, чтобы не тратить лишние параметры на отдельную выходную матрицу.

## 4. Структура проекта

### 4.1 Папка `src/sllm`

Это ядро проекта. Здесь лежат:

- конфиги;
- модель;
- подготовка/чтение бинарных данных;
- утилиты;
- checkpointing.

### 4.2 Папка `scripts`

Это CLI-слой. Каждый скрипт отвечает за отдельный этап пайплайна.

### 4.3 Папка `configs`

Это набор готовых JSON-конфигов:

- архитектура модели;
- смесь pretraining-данных;
- stage 1 pretraining;
- stage 2 annealing;
- конфиг данных для SFT;
- конфиг обучения SFT.

## 5. Модуль `src/sllm/config.py`

Этот модуль хранит типизированные конфиги и функции чтения/записи JSON.

### 5.1 `ModelConfig`

Назначение: описывает архитектуру модели.

Основные поля:

- `vocab_size`: общий размер словаря, включая спецтокены.
- `max_seq_len`: максимальная длина контекста.
- `d_model`: размер скрытого состояния.
- `n_layers`: число transformer blocks.
- `n_heads`: число голов attention.
- `ffn_hidden_dim`: размер промежуточного слоя в `SwiGLU`.
- `rope_theta`: базовый параметр роторных эмбеддингов.
- `rms_norm_eps`: epsilon в `RMSNorm`.
- `initializer_range`: sigma для нормальной инициализации.
- `dropout`: dropout внутри attention.
- `tie_word_embeddings`: связывать ли веса embedding и LM head.
- `bias`: использовать ли bias в линейных слоях.
- `pad_token_id`, `bos_token_id`, `eos_token_id`: идентификаторы спецтокенов.

Основные методы:

- `from_dict(...)`: создаёт dataclass из словаря.
- `to_dict()`: сериализует конфиг обратно в словарь.

### 5.2 `SourceConfig`

Назначение: описывает один источник данных для pretraining.

Основные поля:

- `name`: короткое имя источника.
- `path`: dataset id или путь для `datasets.load_dataset`.
- `split`: split датасета.
- `weight`: вес источника в общей смеси.
- `text_field`: поле с текстом.
- `config_name`: подконфиг датасета, если нужен.
- `revision`: фиксированная ревизия источника.
- `streaming`: загружать ли датасет потоком.
- `shuffle_buffer`: размер буфера при shuffle в streaming-режиме.
- `sample_documents`: ручной лимит документов для tokenizer training.

### 5.3 `DataMixConfig`

Назначение: описывает общую смесь pretraining-данных и правила подготовки токенизатора.

Основные поля:

- `sources`: список `SourceConfig`.
- `tokenizer_sample_documents`: сколько документов брать на обучение токенизатора.
- `tokenizer_min_frequency`: минимальная частота merge-кандидатов.
- `tokenizer_special_tokens`: список спецтокенов.
- `train_tokens`: целевое число train токенов.
- `val_tokens`: целевое число validation токенов.
- `shard_size_tokens`: размер одного бинарного shard.

Основные методы:

- `from_dict(...)`
- `normalized_weights()`: нормализует веса источников до суммы `1.0`.
- `to_dict()`

### 5.4 `TrainConfig`

Назначение: описывает pretraining.

Основные поля:

- пути к данным, логам и checkpoint;
- `seq_len`, `micro_batch_size`, `grad_accum_steps`;
- `max_steps`, `warmup_steps`;
- `learning_rate`, `min_lr`;
- `weight_decay`, `beta1`, `beta2`;
- `grad_clip`;
- `precision`;
- интервалы логирования, оценки и сохранения;
- `compile_model`.

### 5.5 `SFTConfig`

Назначение: описывает supervised fine-tuning.

Структурно похож на `TrainConfig`, но рассчитан на другой режим обучения и другой набор данных.

### 5.6 `load_json(...)`

Назначение: безопасно читает JSON-файл в словарь Python.

### 5.7 `save_json(...)`

Назначение: записывает словарь в JSON с `utf-8` и `indent=2`.

## 6. Модуль `src/sllm/utils.py`

Это набор небольших инфраструктурных функций.

### 6.1 `ensure_dir(path)`

Создает директорию, если ее еще нет, и возвращает `Path`.

Используется почти во всех частях проекта, где нужно сохранить артефакты.

### 6.2 `set_seed(seed)`

Фиксирует seed для:

- `random`
- `numpy`
- `torch`
- всех CUDA-устройств

Нужен для воспроизводимости.

### 6.3 `get_device()`

Возвращает устройство в порядке приоритета:

1. `cuda`
2. `mps`
3. `cpu`

### 6.4 `get_dtype(name)`

Преобразует строковое имя precision в `torch.dtype`.

Поддерживает:

- `bf16`
- `fp16`
- `fp32`

### 6.5 `autocast_context(device, precision)`

Возвращает контекст `torch.autocast` для mixed precision.

Зачем нужен:

- чтобы pretraining и SFT код не дублировали одну и ту же логику выбора precision;
- чтобы на CPU/MPS режим просто отключался.

### 6.6 `format_number(value)`

Форматирует большие числа в человекочитаемом виде:

- `K`
- `M`
- `B`

Используется в логах.

### 6.7 `timestamp()`

Возвращает строку времени для логов.

### 6.8 `model_parameter_count(model)`

Считает общее число параметров модели.

### 6.9 `tokens_per_step(...)`

Считает число токенов на один optimizer step:

`micro_batch_size * grad_accum_steps * seq_len`

Это важно для оценки реального training throughput.

### 6.10 `cosine_lr(...)`

Реализует learning rate scheduler:

- linear warmup;
- затем cosine decay до `min_lr`.

Используется и в pretraining, и в SFT.

### 6.11 `set_optimizer_lr(...)`

Применяет текущий learning rate ко всем параметрическим группам optimizer.

### 6.12 `maybe_enable_tf32(device)`

На CUDA включает TF32 для matmul и cuDNN.

Зачем:

- это часто ускоряет обучение на современных GPU;
- при этом почти всегда приемлемо для такого training pipeline.

### 6.13 `require_cuda_bf16(precision)`

Проверяет, что если выбран `bf16`, то:

- доступна CUDA;
- железо поддерживает BF16.

Иначе кидает понятную ошибку.

### 6.14 `env_int(name, default)`

Читает integer из переменных окружения.

Сейчас почти не используется, но оставлен как маленький helper для будущего расширения.

## 7. Модуль `src/sllm/checkpoint.py`

Отвечает за сохранение и восстановление состояния обучения.

### 7.1 `save_checkpoint(...)`

Сохраняет:

- `step`
- `model.state_dict()`
- `optimizer.state_dict()`
- `model_config`
- `train_config`
- `extra_state`

Зачем такой состав:

- можно продолжать обучение;
- можно выполнять inference без отдельного конфиг-файла;
- можно позже анализировать, с какими гиперпараметрами был получен checkpoint.

### 7.2 `load_checkpoint(...)`

Просто обертка над `torch.load(...)`, которая загружает checkpoint по пути.

## 8. Модуль `src/sllm/model.py`

Это центральный модуль проекта: он реализует саму языковую модель.

### 8.1 `RMSNorm`

Назначение: реализует Root Mean Square Normalization.

Что делает:

- считает средний квадрат по последней размерности;
- нормализует вектор;
- масштабирует результат обучаемым вектором `weight`.

Почему выбран:

- дешевле LayerNorm;
- хорошо работает в современных decoder-only архитектурах.

### 8.2 `rotate_half(x)`

Вспомогательная функция для RoPE.

Что делает:

- разбивает вектор на две половины;
- выполняет поворотную перестановку `(-x2, x1)`.

Это стандартная часть реализации rotary embeddings.

### 8.3 `RotaryEmbedding`

Назначение: заранее строит и кэширует синусы/косинусы для всех позиций до `max_seq_len`.

Поля:

- `cos_cached`
- `sin_cached`

Что делает `forward(...)`:

- по `position_ids` выбирает нужные коэффициенты;
- приводит их к dtype и device входа;
- применяет роторное преобразование к входному тензору.

Почему кэширование важно:

- меньше лишних пересчетов;
- проще и быстрее runtime.

### 8.4 `CausalSelfAttention`

Назначение: реализует masked multi-head self-attention.

Что происходит внутри:

1. линейные проекции в `Q`, `K`, `V`;
2. reshape в форму `[batch, heads, seq, head_dim]`;
3. применение RoPE к `Q` и `K`;
4. вызов `F.scaled_dot_product_attention(...)`;
5. обратный reshape;
6. выходная проекция `o_proj`.

Почему выбран именно `scaled_dot_product_attention`:

- это стандартный путь в современном PyTorch;
- позволяет использовать быстрые CUDA kernels автоматически;
- не добавляет жесткой внешней зависимости на `flash-attn`.

Почему проверяется делимость `d_model % n_heads == 0`:

- без этого attention невозможно корректно разложить на головы.

### 8.5 `SwiGLU`

Назначение: реализует gated MLP.

Внутренние проекции:

- `gate_proj`
- `up_proj`
- `down_proj`

Формула:

`down_proj(silu(gate_proj(x)) * up_proj(x))`

Почему выбран:

- обычно сильнее стандартного GELU-FFN при том же масштабе модели.

### 8.6 `TransformerBlock`

Один блок декодера.

Состав:

- `input_norm`
- `attention`
- `post_attn_norm`
- `mlp`

Схема:

1. pre-norm перед attention;
2. residual connection;
3. pre-norm перед MLP;
4. residual connection.

Это современная и стабильная схема pre-normalization.

### 8.7 `SLLMForCausalLM`

Главный класс модели.

Состав:

- `embed_tokens`
- `layers`
- `norm`
- `lm_head`

Ключевые решения:

- если `tie_word_embeddings=True`, веса `lm_head.weight` связываются с `embed_tokens.weight`;
- линейные и embedding слои инициализируются нормальным распределением с `initializer_range`.

#### Метод `_init_weights(...)`

Отвечает за инициализацию:

- `Linear.weight` и `Embedding.weight` через `Normal(0, sigma)`
- `Linear.bias` обнуляется

#### Метод `forward(...)`

Вход:

- `input_ids`
- `attention_mask`
- `labels`

Логика:

1. проверка длины входа относительно `max_seq_len`;
2. генерация `position_ids`;
3. embedding lookup;
4. последовательный проход по всем `TransformerBlock`;
5. финальная нормализация;
6. вычисление logits через `lm_head`;
7. если есть `labels`, расчет cross-entropy с `ignore_index=-100`.

Возвращает словарь:

- всегда `logits`
- опционально `loss`

Почему `ignore_index=-100`:

- это стандартный способ маскировать токены, которые не должны участвовать в loss;
- активно используется на этапе SFT.

#### Метод `generate(...)`

Базовая autoregressive генерация.

Поддерживает:

- `max_new_tokens`
- `temperature`
- `top_k`
- `eos_token_id`

Логика:

1. берет текущий контекст;
2. считает logits последнего токена;
3. применяет greedy decode или sampling;
4. добавляет новый токен;
5. останавливается по `eos_token_id`, если нужно.

Ограничение:

- здесь нет KV-cache, поэтому длинная генерация менее эффективна, чем могла бы быть.

#### Метод `export_config()`

Возвращает конфиг модели в сериализуемом виде.

## 9. Модуль `src/sllm/data.py`

Этот модуль отвечает за два типа задач:

- запись данных в эффективный бинарный формат;
- чтение и подачу данных в training loop.

### 9.1 `TokenShardWriter`

Назначение: пишет последовательный поток токенов в `.bin`-шарды фиксированного размера.

Поля:

- `output_dir`
- `prefix`
- `shard_size_tokens`
- `buffer`
- `shard_index`
- `shards`

#### `add_tokens(tokens)`

Добавляет список токенов в буфер.

Когда буфер достигает `shard_size_tokens`, записывает один shard на диск.

#### `finalize()`

Дозаписывает остаток буфера и создает manifest JSON.

#### `_write_chunk(chunk)`

Пишет массив токенов как `np.uint16` в файл `.bin` и добавляет запись в manifest.

Почему `uint16`:

- словарь `49152` помещается в 16 бит;
- это в два раза компактнее, чем `int32`.

### 9.2 `SFTShardWriter`

Назначение: потоково пишет подготовленные SFT-примеры в бинарные файлы.

Важное отличие от варианта "копить все в списке":

- здесь примеры пишутся сразу на диск;
- это не упирается в RAM на крупных SFT-датасетах.

#### `add_example(input_ids, labels)`

Пишет один пример:

- `input_ids` как `uint16`
- `labels` как `int32`

Почему `labels` как `int32`:

- потому что они содержат `-100` для masked loss.

#### `finalize()`

Закрывает файловые handle и сохраняет metadata:

- число примеров;
- длина последовательности;
- имена бинарных файлов.

### 9.3 `load_shard_manifest(data_dir, split)`

Назначение: находит manifest-файлы и расширяет записи абсолютными путями к shard-файлам.

Почему это нужно:

- training dataset должен быстро находить физические файлы;
- структура папок может быть либо сразу в split-директории, либо уровнем выше.

### 9.4 `RandomTokenDataset`

Назначение: бесконечно генерирует случайные pretraining-окна из бинарных шардов.

Это `IterableDataset`.

Логика инициализации:

1. загружает manifest;
2. открывает каждый shard как `np.memmap`;
3. считает число доступных стартовых позиций в каждом shard;
4. строит вероятности выбора shard пропорционально его емкости.

Почему это важно:

- большие shard-файлы должны выбираться чаще, чем маленькие;
- sampling остается примерно равномерным по всем токенам корпуса.

#### `__iter__()`

Бесконечный цикл:

1. выбирает shard по вероятностям;
2. выбирает случайный `start`;
3. читает окно длины `seq_len + 1`;
4. формирует:
   - `input_ids = tokens[:-1]`
   - `labels = tokens[1:]`
   - `attention_mask = ones`

Зачем нужен этот класс:

- не нужно держать весь корпус в RAM;
- можно быстро обучать causal LM на огромном наборе данных;
- формат удобен для random-window sampling.

### 9.5 `SequentialEvalDataset`

Назначение: дает детерминированный, последовательный проход по validation shards.

Почему он отдельный:

- для оценки лучше использовать повторяемый и стабильный набор батчей;
- это снижает шум в метриках по сравнению со случайным sampling.

### 9.6 `FixedSFTDataset`

Назначение: читает уже упакованные SFT-данные из memmap-массивов.

В инициализации:

- читает `train_metadata.json` или `val_metadata.json`;
- открывает `input_ids` и `labels` как memmap нужной формы.

#### `__getitem__(index)`

Возвращает:

- `input_ids`
- `labels`
- `attention_mask`

Почему `attention_mask = (input_ids != 0)`:

- `0` используется как `pad_token_id`;
- padded позиции не должны считаться реальными токенами.

## 10. Скрипт `scripts/train_tokenizer.py`

Этот скрипт обучает BPE-токенизатор на миксе источников.

### Основные решения

- используется `tokenizers` вместо медленного Python-only пайплайна;
- используется `ByteLevel` pre-tokenizer и decoder;
- словарь целится в фиксированный размер `49152`, включая спецтокены.

### Основные функции

#### `build_parser()`

Определяет CLI-аргументы:

- `--data-config`
- `--output-dir`
- `--vocab-size`
- `--seed`

#### `iter_source_texts(source, seed, limit)`

Загружает один источник через `datasets.load_dataset(...)` и отдает поток строк.

Особенности:

- поддерживает streaming;
- выполняет shuffle при необходимости;
- обрезает поток по лимиту документов.

#### `mixed_iterator(config, seed)`

Распределяет лимит документов между источниками по их весам и объединяет их в один генератор текстов.

#### `main()`

Главный поток работы:

1. читает `DataMixConfig`;
2. инициализирует `Tokenizer(models.BPE(...))`;
3. обучает его через `BpeTrainer`;
4. находит id спецтокенов;
5. настраивает `TemplateProcessing` для автоматического добавления `<bos>` и `<eos>`;
6. сохраняет `tokenizer.json`;
7. сохраняет `tokenizer_meta.json`.

### Что такое `tokenizer_meta.json`

Это служебный файл с метаданными токенизатора:

- итоговый размер словаря;
- строковые имена спецтокенов;
- их integer id;
- конфиг данных, на котором обучался токенизатор.

Он нужен, чтобы следующие этапы знали:

- какие id использовать для `bos/eos/pad`;
- как корректно делать padding и generation;
- с каким словарем и на каком миксе был создан токенизатор.

## 11. Скрипт `scripts/prepare_pretrain_data.py`

Этот скрипт преобразует сырой текстовый корпус в бинарный токенизированный формат.

Ключевое поведение текущей версии:

- train и val строятся не последовательно по источникам, а как глобальная interleaved смесь;
- доля каждого источника удерживается близкой к target через scheduler по недовыполнению токенного бюджета;
- train и val формируются раздельно, поэтому один и тот же документ не должен одновременно протекать в оба split.

### Основные функции

#### `build_parser()`

CLI для:

- `--data-config`
- `--tokenizer-dir`
- `--output-dir`
- `--seed`

#### `load_tokenizer(tokenizer_dir)`

Загружает:

- `tokenizer.json`
- `tokenizer_meta.json`

#### `iter_source_rows(source, seed)`

Открывает один источник данных и возвращает итератор по строкам датасета.

#### `allocate_token_targets(data_config, total_tokens)`

Вычисляет точное целевое количество токенов на источник.

Особенности:

- сначала считает дробные квоты по весам;
- затем берет `floor`;
- остаток распределяет по источникам с наибольшей дробной частью.

Зачем это нужно:

- сумма target'ов точно совпадает с общим числом токенов;
- нет потерь из-за простого округления вниз.

#### `make_source_state(source, seed)`

Создает runtime-state для одного источника:

- итератор по датасету;
- счетчики использованных документов;
- счетчики train/val токенов;
- флаг `exhausted`.

#### `next_valid_token_ids(state, tokenizer)`

Берет следующий валидный документ из конкретного источника и токенизирует его.

Функция пропускает:

- пустые строки;
- нестроковые записи;
- документы, которые после токенизации дали пустой результат.

В текущей реализации одиночная токенизация вынесена из hot path:

- документы сначала набираются небольшими batch'ами;
- затем токенизируются через `tokenizer.encode_batch(...)`;
- после этого готовые token ids забираются из внутренней очереди источника.

Это заметно уменьшает Python overhead при подготовке большого корпуса.

#### `choose_source_name(states, targets, split, rng)`

Выбирает следующий источник для текущего split.

Логика:

- рассматриваются только источники, у которых еще не выполнен target;
- для каждого считается `progress = written / target`;
- выбирается источник с минимальным progress;
- при равенстве добавляется случайный tie-break.

Идея:

- underrepresented источники получают приоритет;
- смесь остается близкой к заданным весам уже в процессе построения корпуса.

#### `interleave_split(split, writer, states, targets, tokenizer, logger, rng)`

Собирает один глобальный поток токенов для `train` или `val`.

Алгоритм:

1. пока split не набрал target tokens;
2. выбрать следующий источник через `choose_source_name(...)`;
3. взять следующий валидный документ;
4. токенизировать его;
5. обрезать chunk так, чтобы не превысить target источника и split;
6. записать chunk в глобальный `TokenShardWriter`.

Итог:

- внутри split получается общий interleaved токенный поток;
- а не набор отдельных "кусков по источникам".

#### `main()`

Алгоритм:

1. загрузить смесь данных;
2. загрузить токенизатор;
3. вычислить точные train/val token targets по каждому источнику;
4. создать runtime-state и отдельный итератор для каждого источника;
5. interleave-собрать `val`;
6. interleave-собрать `train`;
7. записать глобальные shard-файлы;
8. сохранить `dataset_summary.json`.

### Формат результата

- `data/pretokenized/train/*.bin`
- `data/pretokenized/val/*.bin`
- один глобальный manifest для `train`
- один глобальный manifest для `val`
- общий `dataset_summary.json` с итогами по каждому источнику

## 12. Скрипт `scripts/train_pretrain.py`

Это основной training loop для предобучения.

### Основные решения

- training идет через `RandomTokenDataset`;
- optimizer: `AdamW`;
- scheduler: warmup + cosine decay;
- mixed precision: `bf16` или `fp16`;
- gradient accumulation поддерживает эффективный глобальный batch;
- checkpoint сохраняет и веса, и optimizer state.

### Основные функции

#### `build_parser()`

CLI:

- `--model-config`
- `--train-config`
- `--max-steps`

`--max-steps` нужен для dry run и локальной отладки.

#### `build_optimizer(model, config, device)`

Разделяет параметры на две группы:

- с weight decay;
- без weight decay.

Без decay идут:

- bias;
- вектора/скаляры нормализации;
- одномерные параметры.

Если устройство `cuda`, включает fused AdamW.

#### `evaluate(model, config, device)`

Выполняет validation:

1. создает `SequentialEvalDataset`;
2. собирает `DataLoader`;
3. считает средний loss;
4. преобразует его в perplexity.

#### `maybe_load_weights(model, optimizer, config, device)`

Решает два разных сценария:

- `init_from`: загрузить только веса и начать новый training run;
- `resume_from`: загрузить и веса, и optimizer state, и шаг.

Это важное разделение: оно делает pipeline удобным и для transfer, и для продолжения обучения.

#### `save_run_config(output_dir, model_config, train_config)`

Сохраняет эффективный конфиг конкретного запуска.

#### `main()`

Главный training loop:

1. загрузить конфиги;
2. установить seed;
3. выбрать устройство и precision;
4. создать output/checkpoint директории;
5. собрать `RandomTokenDataset` и `DataLoader`;
6. построить модель;
7. при необходимости включить `torch.compile`;
8. создать optimizer и scaler;
9. загрузить checkpoint, если нужно;
10. на каждом шаге:
    - вычислить текущий `lr`;
    - накопить градиенты за `grad_accum_steps`;
    - применить gradient clipping;
    - сделать `optimizer.step()`;
    - залогировать loss и throughput;
    - периодически оценить val loss;
    - периодически сохранить checkpoint.

### Почему `compile_model` по умолчанию выключен

Потому что безопасный default важнее теоретического ускорения:

- `torch.compile` может давать долгую первую компиляцию;
- иногда осложняет отладку;
- иногда ведет себя нестабильно при определенных версиях PyTorch/CUDA.

После dry run его можно включить, если окружение работает стабильно.

## 13. Скрипт `scripts/prepare_sft_data.py`

Этот скрипт превращает instruction/chat датасет в фиксированные обучающие примеры для SFT.

### Основные решения

- поддерживается несколько форматов входного датасета;
- loss считается только на ответах assistant;
- данные потоково пишутся на диск;
- результат хранится в memmap-friendly бинарном формате.

### Основные функции

#### `build_parser()`

CLI:

- `--config`
- `--tokenizer-dir`
- `--output-dir`
- `--seq-len`
- `--seed`

#### `load_tokenizer(tokenizer_dir)`

Загружает токенизатор и его метаданные.

#### `row_to_messages(row, config)`

Нормализует строку датасета в единый внутренний формат:

`list[{"role": ..., "content": ...}]`

Поддерживаемые схемы:

- `messages`
- `prompt_response`
- `alpaca`

Зачем это сделано:

- можно использовать разные SFT-датасеты без переписывания основного кода;
- формат внутри пайплайна остается единым.

#### `tokenize_messages(tokenizer, messages, bos_id, eos_id)`

Преобразует диалог в две последовательности:

- `input_ids`
- `labels`

Ключевая идея:

- токены `assistant` участвуют в loss;
- токены `system/user` получают `-100` и исключаются из loss.

Это одна из важнейших функций проекта, потому что она задает саму постановку SFT-задачи.

#### `pad_or_truncate(input_ids, labels, seq_len, pad_id)`

Приводит пример к фиксированной длине:

- слишком длинное обрезается;
- короткое pad'ится;
- pad-позиции маскируются в `labels` через `-100`.

#### `main()`

Алгоритм:

1. загрузить конфиг и токенизатор;
2. определить id спецтокенов;
3. загрузить SFT-датасет;
4. при необходимости перемешать;
5. первые `val_examples` отправить в validation;
6. остальные отправить в train;
7. сохранить metadata и summary.

## 14. Скрипт `scripts/train_sft.py`

Почти тот же training loop, что и pretraining, но адаптирован под конечный датасет `FixedSFTDataset`.

### Главное отличие от pretraining

- источник данных не бесконечный случайный, а конечный индексируемый;
- validation идет по уже собранному `val` split;
- `labels` уже частично замаскированы на этапе подготовки данных.

### Основные функции

#### `build_optimizer(...)`

Повторяет логику pretraining-скрипта: разделяет decay/no_decay группы.

#### `evaluate(...)`

Считает средний validation loss и perplexity по SFT validation set.

#### `save_run_config(...)`

Сохраняет effective run config.

#### `main()`

Алгоритм:

1. загрузить `ModelConfig` и `SFTConfig`;
2. инициализировать устройство;
3. открыть `FixedSFTDataset` для `train` и `val`;
4. построить модель;
5. загрузить pretrain checkpoint или resume checkpoint;
6. обучать с gradient accumulation;
7. логировать loss;
8. периодически валидировать и сохранять checkpoint.

## 15. Скрипт `scripts/eval_perplexity.py`

Это небольшой utility-скрипт для измерения validation perplexity на pretraining shards.

### Основные функции

#### `build_parser()`

CLI:

- `--checkpoint`
- `--model-config`
- `--data-dir`
- `--seq-len`
- `--batch-size`
- `--batches`
- `--precision`

#### `main()`

Алгоритм:

1. загрузить checkpoint;
2. восстановить конфиг модели;
3. построить модель;
4. создать `SequentialEvalDataset`;
5. посчитать средний loss;
6. вывести `val_loss` и `perplexity`.

## 16. Скрипт `scripts/generate.py`

Нужен для быстрой ручной проверки итоговой модели.

### Основные функции

#### `build_parser()`

CLI:

- `--checkpoint`
- `--tokenizer-dir`
- `--prompt`
- `--max-new-tokens`
- `--temperature`
- `--top-k`
- `--model-config`

#### `main()`

Алгоритм:

1. загрузить токенизатор и `tokenizer_meta.json`;
2. взять `bos_token_id`;
3. загрузить checkpoint и конфиг модели;
4. построить модель;
5. токенизировать prompt;
6. вызвать `model.generate(...)`;
7. декодировать и вывести результат.

## 17. Конфиги в `configs/`

### 17.1 `model_70m.json`

Описывает архитектуру модели.

### 17.2 `data_mix_10b.json`

Описывает смесь источников:

- FineWeb-Edu
- Cosmopedia v2
- The Stack Python
- FineMath

Также задает:

- число токенов для train/val;
- размер shard;
- параметры выборки для tokenizer training.

### 17.3 `pretrain_5090_stage1.json`

Конфиг основного pretraining:

- `seq_len=2048`
- относительно крупный effective batch через gradient accumulation
- высокий learning rate для маленькой модели

### 17.4 `pretrain_5090_stage2_anneal.json`

Конфиг второй стадии:

- `seq_len=8192`
- меньше micro batch;
- ниже learning rate;
- инициализация из stage 1 checkpoint.

Зачем отдельная стадия:

- длинный контекст сильно дороже по памяти;
- разумнее сначала научить модель на `2048`, а затем дообучить на `8192`.

### 17.5 `sft_data_smoltalk.json`

Описывает SFT dataset и его формат.

### 17.6 `sft_5090.json`

Описывает SFT training run:

- путь к подготовленным данным;
- путь к стартовому checkpoint;
- hyperparameters обучения.

## 18. Форматы файлов и артефактов

### 18.1 Токенизатор

Файлы:

- `tokenizer.json`
- `tokenizer_meta.json`
- `tokenizer_summary.json`

### 18.2 Pretokenized data

Файлы:

- `*.bin` с `uint16` токенами;
- `*_manifest.json` с описанием shard-файлов;
- `dataset_summary.json`.

### 18.3 SFT data

Файлы:

- `train_input_ids.bin`
- `train_labels.bin`
- `train_metadata.json`
- `val_input_ids.bin`
- `val_labels.bin`
- `val_metadata.json`

### 18.4 Checkpoints

Checkpoint содержит:

- веса модели;
- состояние optimizer;
- шаг обучения;
- конфиги;
- дополнительное состояние, например `tokens_seen`.

## 19. Главные проектные компромиссы

### 19.1 Что упрощено сознательно

- нет DDP/FSDP/ZeRO;
- нет KV-cache в генерации;
- нет packed SFT из нескольких коротких диалогов в один sequence;
- нет внешней зависимости на `flash-attn`;
- нет Hugging Face `Trainer` и сложной training framework abstraction.

### 19.2 Почему это нормально

Цель проекта не в том, чтобы построить максимально абстрактный фреймворк, а в том, чтобы получить:

- прозрачный код;
- понятный pipeline;
- минимум скрытой магии;
- удобство локального запуска и модификации.

## 20. Как связаны между собой модули

Связи такие:

- `config.py` используется почти всеми скриптами;
- `utils.py` используется training-скриптами и checkpointing;
- `model.py` используется в pretraining, SFT, eval и generation;
- `data.py` используется в подготовке pretraining-данных, подготовке SFT-данных, pretraining и SFT;
- `checkpoint.py` используется в `train_pretrain.py`, `train_sft.py`, `eval_perplexity.py`, `generate.py`.

То есть проект построен вокруг пяти базовых доменных сущностей:

1. конфиг;
2. токенизатор;
3. модель;
4. бинарные данные;
5. checkpoint.

## 21. Как читать проект в правильном порядке

Если программисту нужно быстро понять проект, лучший порядок чтения такой:

1. `README.md`
2. `configs/model_70m.json`
3. `src/sllm/config.py`
4. `src/sllm/model.py`
5. `src/sllm/data.py`
6. `scripts/train_tokenizer.py`
7. `scripts/prepare_pretrain_data.py`
8. `scripts/train_pretrain.py`
9. `scripts/prepare_sft_data.py`
10. `scripts/train_sft.py`

## 22. Итог

Этот проект реализует компактный, но полный стек обучения небольшой decoder-only языковой модели:

- с современной архитектурой;
- с практичным форматом данных;
- с воспроизводимыми конфигами;
- с разделением pretraining и SFT;
- с минимальной инфраструктурной сложностью.

Главная идея проекта: вместо тяжелого framework-first подхода использовать прозрачный code-first pipeline, который легко читать, менять и запускать на одной мощной GPU.
