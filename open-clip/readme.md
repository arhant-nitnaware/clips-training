### Test train on NWPU-Captions for 1 epoch
python -m open_clip_train.main \
    --train-data ../../dataset/NWPU-Captions/train.csv \
    --val-data ../../dataset/NWPU-Captions/val.csv \
    --dataset-type csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --csv-separator "," \
    --model ViT-B-32 \
    --pretrained openai \
    --batch-size 32 \
    --workers 8 \
    --epochs 1 \
    --lr 1e-5 \
    --precision amp \
    --logs logs


# OpenCLIP Training Guide

This repository supports training:

* CLIP models
* Distilled CLIP models (TinyCLIP-style)

Distillation is enabled using:

```bash
--distill-model <teacher_model>
--distill-pretrained <teacher_checkpoint>
```

---

# Requirements

* Python 3.10+
* PyTorch
* CUDA (recommended)
* OpenCLIP training framework

---

# Dataset Format

Training is performed from a CSV.

Example:

```text
filepath,caption
images/img1.jpg,A satellite image of an airport.
images/img2.jpg,A residential area.
...
```

---

# Directory Structure

```text
dataset/
└── NWPU-Captions/
    ├── train.csv
    ├── val.csv
    └── images/
```

---

# Training

## Standard CLIP

### open_clip_train.main usage
```bash
python -m open_clip_train.main --help 
usage: main.py [-h] [--train-data TRAIN_DATA]
               [--train-data-upsampling-factors TRAIN_DATA_UPSAMPLING_FACTORS]
               [--val-data VAL_DATA] [--train-num-samples TRAIN_NUM_SAMPLES]
               [--val-num-samples VAL_NUM_SAMPLES]
               [--dataset-type {webdataset,webdataset-audio,csv,synthetic,synthetic-audio,auto}]
               [--audio-ext AUDIO_EXT] [--pretrained-audio PRETRAINED_AUDIO]
               [--audio-fill {pad,repeat,repeatpad}]
               [--audio-trunc {rand_trunc,trunc,fusion}] [--audio-fusion]
               [--audio-int16-normalize]
               [--audio-zeroshot-dataset AUDIO_ZEROSHOT_DATASET]
               [--audio-zeroshot-split AUDIO_ZEROSHOT_SPLIT]
               [--audio-zeroshot-audio-key AUDIO_ZEROSHOT_AUDIO_KEY]
               [--audio-zeroshot-target-key AUDIO_ZEROSHOT_TARGET_KEY]
               [--audio-zeroshot-class-key AUDIO_ZEROSHOT_CLASS_KEY]
               [--audio-zeroshot-template AUDIO_ZEROSHOT_TEMPLATES]
               [--audio-zeroshot-workers AUDIO_ZEROSHOT_WORKERS]
               [--audio-zeroshot-multiprocessing-context {fork,forkserver,spawn}]
               [--audio-multiprocessing-context {fork,forkserver,spawn}]
               [--dataset-resampled] [--csv-separator CSV_SEPARATOR]
               [--csv-img-key CSV_IMG_KEY] [--csv-caption-key CSV_CAPTION_KEY]
               [--image-key IMAGE_KEY] [--max-image-pixels MAX_IMAGE_PIXELS]
               [--text-key TEXT_KEY] [--json-text-key JSON_TEXT_KEY]
               [--json-text-key-probs JSON_TEXT_KEY_PROBS [JSON_TEXT_KEY_PROBS ...]]
               [--imagenet-val IMAGENET_VAL] [--imagenet-v2 IMAGENET_V2]
               [--cache-dir CACHE_DIR] [--logs LOGS] [--log-local] [--name NAME]
               [--workers WORKERS] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
               [--epochs-cooldown EPOCHS_COOLDOWN] [--lr LR] [--beta1 BETA1]
               [--beta2 BETA2] [--eps EPS] [--wd WD] [--momentum MOMENTUM]
               [--warmup WARMUP] [--opt OPT] [--opt-kwargs [OPT_KWARGS ...]]
               [--opt-fallback-list [PATTERN ...]] [--text-layer-decay TEXT_LAYER_DECAY]
               [--image-layer-decay IMAGE_LAYER_DECAY]
               [--audio-layer-decay AUDIO_LAYER_DECAY] [--wd-exclude [PATTERN ...]]
               [--text-pooler-own-group] [--use-bn-sync] [--skip-scheduler]
               [--lr-scheduler LR_SCHEDULER] [--lr-cooldown-end LR_COOLDOWN_END]
               [--lr-cooldown-power LR_COOLDOWN_POWER] [--save-frequency SAVE_FREQUENCY]
               [--save-most-recent] [--zeroshot-frequency ZEROSHOT_FREQUENCY]
               [--val-frequency VAL_FREQUENCY]
               [--val-retrieval-chunk-size VAL_RETRIEVAL_CHUNK_SIZE]
               [--val-retrieval-precision {fp32,model}] [--resume RESUME]
               [--precision {amp,amp_bf16,amp_bfloat16,bf16,fp16,pure_bf16,pure_fp16,fp32}]
               [--model MODEL] [--pretrained PRETRAINED] [--pretrained-image]
               [--lock-image] [--lock-image-unlocked-groups LOCK_IMAGE_UNLOCKED_GROUPS]
               [--lock-image-freeze-bn-stats] [--image-mean MEAN [MEAN ...]]
               [--image-std STD [STD ...]]
               [--image-interpolation {bicubic,bilinear,random}]
               [--image-resize-mode {shortest,longest,squash}] [--aug-cfg [AUG_CFG ...]]
               [--grad-checkpointing] [--local-loss] [--gather-with-grad]
               [--force-context-length FORCE_CONTEXT_LENGTH]
               [--force-image-size FORCE_IMAGE_SIZE [FORCE_IMAGE_SIZE ...]]
               [--force-quick-gelu] [--force-patch-dropout FORCE_PATCH_DROPOUT]
               [--force-custom-text] [--torchcompile]
               [--torchcompile-strategy {model,task,step}]
               [--torchcompile-backend TORCHCOMPILE_BACKEND]
               [--torchcompile-mode TORCHCOMPILE_MODE] [--accum-freq ACCUM_FREQ]
               [--device DEVICE] [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
               [--report-to REPORT_TO] [--wandb-notes WANDB_NOTES]
               [--wandb-project-name WANDB_PROJECT_NAME] [--debug] [--copy-codebase]
               [--ddp-static-graph] [--fsdp] [--fsdp-no-reshard-after-forward]
               [--fsdp-offload-cpu] [--fsdp-checkpoint {full,sharded}]
               [--no-set-device-rank] [--seed SEED] [--grad-clip-norm GRAD_CLIP_NORM]
               [--lock-text] [--lock-text-unlocked-layers LOCK_TEXT_UNLOCKED_LAYERS]
               [--lock-text-freeze-layer-norm] [--log-every-n-steps LOG_EVERY_N_STEPS]
               [--log-metric-every-n-steps LOG_METRIC_EVERY_N_STEPS]
               [--train-loss-ema-samples TRAIN_LOSS_EMA_SAMPLES]
               [--coca-caption-loss-weight COCA_CAPTION_LOSS_WEIGHT]
               [--coca-contrastive-loss-weight COCA_CONTRASTIVE_LOSS_WEIGHT]
               [--remote-sync REMOTE_SYNC]
               [--remote-sync-frequency REMOTE_SYNC_FREQUENCY]
               [--remote-sync-protocol {s3,fsspec}] [--delete-previous-checkpoint]
               [--distill-model DISTILL_MODEL] [--distill-pretrained DISTILL_PRETRAINED]
               [--use-bnb-linear USE_BNB_LINEAR] [--siglip]
               [--loss-dist-impl LOSS_DIST_IMPL] [--use-naflex] [--force-naflex-vision]
               [--naflex-num-train-image-tokens NAFLEX_NUM_TRAIN_IMAGE_TOKENS]
               [--naflex-patch-sizes NAFLEX_PATCH_SIZES [NAFLEX_PATCH_SIZES ...]]
               [--naflex-patch-size-probs NAFLEX_PATCH_SIZE_PROBS [NAFLEX_PATCH_SIZE_PROBS ...]]
               [--naflex-seq-lens NAFLEX_SEQ_LENS [NAFLEX_SEQ_LENS ...]]
               [--naflex-seq-len-probs NAFLEX_SEQ_LEN_PROBS [NAFLEX_SEQ_LEN_PROBS ...]]
               [--naflex-max-tokens-per-batch NAFLEX_MAX_TOKENS_PER_BATCH]
               [--naflex-max-text-tokens NAFLEX_MAX_TEXT_TOKENS]
               [--naflex-batch-divisor NAFLEX_BATCH_DIVISOR]
               [--naflex-loss-scale {none,linear,sqrt}] [--length-bucketing]
               [--bucket-pool BUCKET_POOL] [--bucket-chunk BUCKET_CHUNK]
               [--bucket-prefetch-pools BUCKET_PREFETCH_POOLS]
               [--naflex-pad-multiple NAFLEX_PAD_MULTIPLE]
               [--text-pad-multiple TEXT_PAD_MULTIPLE]

options:
  -h, --help            show this help message and exit
  --train-data TRAIN_DATA
                        Path to file(s) with training data. When using webdataset,
                        multiple datasources can be combined using the `::` separator.
  --train-data-upsampling-factors TRAIN_DATA_UPSAMPLING_FACTORS
                        When using multiple data sources with webdataset and sampling
                        with replacement, this can be used to upsample specific data
                        sources. Similar to --train-data, this should be a string with
                        as many numbers as there are data sources, separated by `::`
                        (e.g. 1::2::0.5) By default, datapoints are sampled uniformly
                        regardless of the dataset sizes.
  --val-data VAL_DATA   Path to file(s) with validation data
  --train-num-samples TRAIN_NUM_SAMPLES
                        Number of samples in dataset. Required for webdataset if not
                        available in info file.
  --val-num-samples VAL_NUM_SAMPLES
                        Number of samples in dataset. Useful for webdataset if not
                        available in info file.
  --dataset-type {webdataset,webdataset-audio,csv,synthetic,synthetic-audio,auto}
                        Which type of dataset to process.
  --audio-ext AUDIO_EXT
                        Audio file extension in WebDataset shards (wav, flac, mp3, ogg).
  --pretrained-audio PRETRAINED_AUDIO
                        Path to pretrained audio encoder checkpoint.
  --audio-fill {pad,repeat,repeatpad}
                        How to fill audio shorter than clip_samples.
  --audio-trunc {rand_trunc,trunc,fusion}
                        How to truncate audio longer than clip_samples.
  --audio-fusion        Enable HTSAT fusion preprocessing for longer audio clips.
  --audio-int16-normalize
                        Apply int16 quantization normalization in audio preprocessing.
  --audio-zeroshot-dataset AUDIO_ZEROSHOT_DATASET
                        Hugging Face audio classification dataset for CLAP zero-shot
                        eval, e.g. ashraq/esc50.
  --audio-zeroshot-split AUDIO_ZEROSHOT_SPLIT
                        Dataset split for audio zero-shot eval.
  --audio-zeroshot-audio-key AUDIO_ZEROSHOT_AUDIO_KEY
                        Audio column name for audio zero-shot eval.
  --audio-zeroshot-target-key AUDIO_ZEROSHOT_TARGET_KEY
                        Target column name for audio zero-shot eval.
  --audio-zeroshot-class-key AUDIO_ZEROSHOT_CLASS_KEY
                        Optional class-name column for audio zero-shot eval.
  --audio-zeroshot-template AUDIO_ZEROSHOT_TEMPLATES
                        Prompt template for audio zero-shot eval. May be passed multiple
                        times; must contain {}.
  --audio-zeroshot-workers AUDIO_ZEROSHOT_WORKERS
                        DataLoader workers for Hugging Face audio zero-shot eval.
                        Defaults to 0 until multiprocessing contexts are tested more
                        broadly.
  --audio-zeroshot-multiprocessing-context {fork,forkserver,spawn}
                        Multiprocessing context for audio zero-shot DataLoader workers.
  --audio-multiprocessing-context {fork,forkserver,spawn}
                        Multiprocessing context for the (training/eval) audio DataLoader
                        workers. forkserver avoids the fork-after-torchaudio-threads
                        deadlock; only applied when --workers > 0.
  --dataset-resampled   Whether to use sampling with replacement for webdataset shard
                        selection.
  --csv-separator CSV_SEPARATOR
                        For csv-like datasets, which separator to use.
  --csv-img-key CSV_IMG_KEY
                        For csv-like datasets, the name of the key for the image paths.
  --csv-caption-key CSV_CAPTION_KEY
                        For csv-like datasets, the name of the key for the captions.
  --image-key IMAGE_KEY
                        For image WebDataset datasets, the tar member suffix holding the
                        image. Accepts ';'-separated alternatives, e.g.
                        'jpg;png;jpeg;webp'.
  --max-image-pixels MAX_IMAGE_PIXELS
                        Drop WebDataset images whose width*height exceeds this, checked
                        from the header before the costly decode. Default 25M px; 0
                        disables.
  --text-key TEXT_KEY   For WebDataset datasets, the tar member suffix holding the
                        caption text (default 'txt'). Accepts ';'-separated
                        alternatives, e.g. 'txt;caption'. Ignored when --json-text-key
                        is set.
  --json-text-key JSON_TEXT_KEY
                        For WebDataset datasets, read the caption from this field of
                        each sample's .json member instead of a text file. Accepts
                        ';'-separated priority alternatives, e.g.
                        'caption_florence-2-large;caption_sharegpt4v-7b'. Takes
                        precedence over --text-key. Without --json-text-key-probs the
                        first non-empty key in order wins.
  --json-text-key-probs JSON_TEXT_KEY_PROBS [JSON_TEXT_KEY_PROBS ...]
                        Sampling probabilities for --json-text-key caption keys, in the
                        same order. Captions are drawn into a random priority order by
                        these weights, then the first non-empty wins.
                        Unspecified/trailing keys default to 0 (fallback only). Need not
                        sum to 1.
  --imagenet-val IMAGENET_VAL
                        Path to imagenet val set for conducting zero shot evaluation.
  --imagenet-v2 IMAGENET_V2
                        Path to imagenet v2 for conducting zero shot evaluation.
  --cache-dir CACHE_DIR
                        Override system default cache path for model & tokenizer file
                        downloads.
  --logs LOGS           Where to store tensorboard logs. Use None to avoid storing logs.
  --log-local           log files on local master, otherwise global master only.
  --name NAME           Optional identifier for the experiment when storing logs.
                        Otherwise use current time.
  --workers WORKERS     Number of dataloader workers per GPU.
  --batch-size BATCH_SIZE
                        Batch size per GPU. Ignored for NaFlex WebDataset training.
  --epochs EPOCHS       Number of epochs to train for.
  --epochs-cooldown EPOCHS_COOLDOWN
                        When scheduler w/ cooldown used, perform cooldown from
                        total_epochs - cooldown_epochs onwards.
  --lr LR               Learning rate.
  --beta1 BETA1         Adam beta 1.
  --beta2 BETA2         Adam beta 2.
  --eps EPS             Adam epsilon.
  --wd WD               Weight decay.
  --momentum MOMENTUM   Momentum (for timm optimizers).
  --warmup WARMUP       Number of steps to warmup for.
  --opt OPT             Which optimizer to use. Choices are ['adamw', 'nadamw' (torch
                        NAdam w/ decoupled weight decay), or any timm optimizer
                        'timm/{opt_name}'].
  --opt-kwargs [OPT_KWARGS ...]
                        Additional optimizer keyword arguments, passed as key=value
                        pairs. The fallback LR scale for Muon-family optimizers goes
                        here (e.g. fallback_lr_scale=0.5).
  --opt-fallback-list [PATTERN ...]
                        Param-name glob patterns routed to a hybrid optimizer's fallback
                        (Muon-family timm opts only, e.g. timm/nadamuon): matched params
                        use the AdamW fallback instead of Muon. No effect for non-Muon
                        optimizers; invalid for torch optimizers. e.g. --opt-fallback-
                        list 'text.transformer.embeddings.*' '*.proj.*'.
  --text-layer-decay TEXT_LAYER_DECAY
                        Layer-wise LR decay for the text tower: lr(group) = lr *
                        decay**(depth_from_head). Off when unset (or 1.0). A gentle
                        alternative to freezing a pretrained text encoder (e.g. 0.65).
  --image-layer-decay, --visual-layer-decay IMAGE_LAYER_DECAY
                        Layer-wise LR decay for the image tower (builtin ViT/ResNet or
                        timm trunk): lr(group) = lr * decay**(depth_from_head); the
                        projection/adapter head stays at full LR. Off when unset (or
                        1.0).
  --audio-layer-decay AUDIO_LAYER_DECAY
                        Layer-wise LR decay for the audio tower (model.audio, e.g. a
                        NaFlex spectrogram-ViT in CLAP): same lr *
                        decay**(depth_from_head) rule. Off when unset (or 1.0). Note: a
                        from-scratch audio tower usually wants full LR (leave off); this
                        is for fine-tuning a pretrained audio encoder.
  --wd-exclude [PATTERN ...]
                        Extra parameter-name glob patterns whose params skip weight
                        decay, on top of the default rule (1-D params + the model's
                        no_weight_decay()). Matched against full param names with
                        fnmatch, so use '*' for substrings, e.g. --wd-exclude '*.bias'
                        'visual.proj*' '*pos_embed*'.
  --text-pooler-own-group
                        Give the text readout pooler its own layer-wise-LR-decay / lock
                        group, one step below the projection head. Default (flag
                        absent): fold the pooler into the projection head (full LR).
  --use-bn-sync         Whether to use batch norm sync.
  --skip-scheduler      Use this flag to skip the learning rate decay.
  --lr-scheduler LR_SCHEDULER
                        LR scheduler. One of: 'cosine', 'const' (constant), 'const-
                        cooldown' (constant w/ cooldown). Default: cosine
  --lr-cooldown-end LR_COOLDOWN_END
                        End learning rate for cooldown schedule. Default: 0
  --lr-cooldown-power LR_COOLDOWN_POWER
                        Power for polynomial cooldown schedule. Default: 1.0 (linear
                        decay)
  --save-frequency SAVE_FREQUENCY
                        How often to save checkpoints.
  --save-most-recent    Always save the most recent model trained to epoch_latest.pt.
  --zeroshot-frequency ZEROSHOT_FREQUENCY
                        How often to run zero shot.
  --val-frequency VAL_FREQUENCY
                        How often to run evaluation with val data.
  --val-retrieval-chunk-size VAL_RETRIEVAL_CHUNK_SIZE
                        Chunk size for exact validation retrieval metrics. Smaller
                        values reduce peak score-matrix memory; set 0 to score the full
                        matrix in one CPU block.
  --val-retrieval-precision {fp32,model}
                        Precision for validation retrieval scoring. 'fp32' casts score
                        chunks to float32; 'model' keeps the feature tensor dtype.
  --resume RESUME       path to latest checkpoint (default: none)
  --precision {amp,amp_bf16,amp_bfloat16,bf16,fp16,pure_bf16,pure_fp16,fp32}
                        Floating point precision.
  --model MODEL         Name of the vision backbone to use.
  --pretrained PRETRAINED
                        Use a pretrained CLIP model weights with the specified tag or
                        file path.
  --pretrained-image    Load imagenet pretrained weights for image tower backbone if
                        available.
  --lock-image          Lock full image tower by disabling gradients.
  --lock-image-unlocked-groups LOCK_IMAGE_UNLOCKED_GROUPS
                        Leave last n image tower layer groups unlocked.
  --lock-image-freeze-bn-stats
                        Freeze BatchNorm running stats in image tower for any locked
                        layers.
  --image-mean MEAN [MEAN ...]
                        Override default image mean value of dataset
  --image-std STD [STD ...]
                        Override default image std deviation of of dataset
  --image-interpolation {bicubic,bilinear,random}
                        Override default image resize interpolation
  --image-resize-mode {shortest,longest,squash}
                        Override default image resize (& crop) mode during inference
  --aug-cfg [AUG_CFG ...]
  --grad-checkpointing  Enable gradient checkpointing.
  --local-loss          calculate loss w/ local features @ global (instead of realizing
                        full global @ global matrix)
  --gather-with-grad    enable full distributed gradient for feature gather
  --force-context-length FORCE_CONTEXT_LENGTH
                        Override default context length
  --force-image-size FORCE_IMAGE_SIZE [FORCE_IMAGE_SIZE ...]
                        Override default image size
  --force-quick-gelu    Force use of QuickGELU activation for non-OpenAI transformer
                        models.
  --force-patch-dropout FORCE_PATCH_DROPOUT
                        Override the patch dropout during training, for fine tuning with
                        no dropout near the end as in the paper
  --force-custom-text   Force use of CustomTextCLIP model (separate text-tower).
  --torchcompile        Enable torch.compile, requires pytorch 2.0 or later.
  --torchcompile-strategy {model,task,step}
                        Compile strategy when --torchcompile is enabled: 'model'
                        compiles trainable_module before distributed wrapping, 'task'
                        compiles task train/eval forward callables, 'step' compiles the
                        single-batch forward/backward/optimizer step.
  --torchcompile-backend TORCHCOMPILE_BACKEND
                        Optional torch.compile backend, e.g. inductor or eager.
  --torchcompile-mode TORCHCOMPILE_MODE
                        Optional torch.compile mode, e.g. default, reduce-overhead, or
                        max-autotune.
  --accum-freq ACCUM_FREQ
                        Update the model every --acum-freq steps.
  --device DEVICE       Accelerator to use.
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend. "nccl" for GPU, "hccl" for Ascend NPU
  --report-to REPORT_TO
                        Comma-separated logging backends: 'wandb', 'trackio' (local-
                        first, wandb-compatible), 'tensorboard'. 'wandb' and 'trackio'
                        are mutually exclusive; either can be combined with
                        'tensorboard'.
  --wandb-notes WANDB_NOTES
                        Notes if logging with wandb
  --wandb-project-name WANDB_PROJECT_NAME
                        Name of the project if logging with wandb.
  --debug               If true, more information is logged.
  --copy-codebase       If true, we copy the entire base on the log directory, and
                        execute from there.
  --ddp-static-graph    Enable static graph optimization for DDP in PyTorch >= 1.11.
  --fsdp                Use FSDP2 (fully_shard) instead of DDP for distributed training.
  --fsdp-no-reshard-after-forward
                        Disable resharding parameters after forward pass. Lower
                        communication but higher memory.
  --fsdp-offload-cpu    Offload FSDP parameters to CPU when not in use.
  --fsdp-checkpoint {full,sharded}
                        FSDP checkpoint type. 'full' gathers to rank 0 as a single .pt
                        file. 'sharded' uses DCP per-rank shards in a directory (faster,
                        lower memory).
  --no-set-device-rank  Don't set device index from local rank (when
                        CUDA_VISIBLE_DEVICES restricted to one per proc).
  --seed SEED           Default random seed.
  --grad-clip-norm GRAD_CLIP_NORM
                        Gradient clip.
  --lock-text           Lock full text tower by disabling gradients.
  --lock-text-unlocked-layers LOCK_TEXT_UNLOCKED_LAYERS
                        Leave last n text tower layer groups unlocked.
  --lock-text-freeze-layer-norm
                        Freeze LayerNorm running stats in text tower for any locked
                        layers.
  --log-every-n-steps LOG_EVERY_N_STEPS
                        Log every n steps to the console (the human-readable line).
  --log-metric-every-n-steps LOG_METRIC_EVERY_N_STEPS
                        Log scalars to tensorboard/wandb every n steps (denser than the
                        console for smooth curves). Set 1 to log every step. The loss is
                        all-reduced across ranks here so the logged value is the true
                        global-batch loss (under --local-loss each rank only sees a
                        1/world_size slice).
  --train-loss-ema-samples TRAIN_LOSS_EMA_SAMPLES
                        Smoothing horizon (in samples) for the console loss EMA shown in
                        parentheses. Robust to batch size / accum / world size / NaFlex
                        packing. 0 disables it (console parentheses revert to the epoch
                        running average).
  --coca-caption-loss-weight COCA_CAPTION_LOSS_WEIGHT
                        Weight assigned to caption loss in CoCa.
  --coca-contrastive-loss-weight COCA_CONTRASTIVE_LOSS_WEIGHT
                        Weight assigned to contrastive loss when training CoCa.
  --remote-sync REMOTE_SYNC
                        Optinoally sync with a remote path specified by this arg
  --remote-sync-frequency REMOTE_SYNC_FREQUENCY
                        How frequently to sync to a remote directly if --remote-sync is
                        not None.
  --remote-sync-protocol {s3,fsspec}
                        How to do the remote sync backup if --remote-sync is not None.
  --delete-previous-checkpoint
                        If true, delete previous checkpoint after storing a new one.
  --distill-model DISTILL_MODEL
                        Which model arch to distill from, if any.
  --distill-pretrained DISTILL_PRETRAINED
                        Which pre-trained weights to distill from, if any.
  --use-bnb-linear USE_BNB_LINEAR
                        Replace the network linear layers from the bitsandbytes library.
                        Allows int8 training/inference, etc.
  --siglip              Use SigLip (sigmoid) loss.
  --loss-dist-impl LOSS_DIST_IMPL
                        A string to specify a specific distributed loss implementation.
  --use-naflex          Use NaFlex WebDataset batching for training and NaFlex
                        patchified validation / zero-shot loaders.
  --force-naflex-vision
                        Convert compatible timm EVA/ViT vision towers to NaFlexViT
                        without enabling the NaFlex data pipeline. --use-naflex implies
                        this.
  --naflex-num-train-image-tokens NAFLEX_NUM_TRAIN_IMAGE_TOKENS
                        Number of image tokens per training epoch for NaFlex schedule
                        creation. Use this instead of --train-num-samples to target a
                        token budget.
  --naflex-patch-sizes NAFLEX_PATCH_SIZES [NAFLEX_PATCH_SIZES ...]
                        Patch sizes to sample for NaFlex training. Eval uses the first
                        value. Defaults to 16 when omitted.
  --naflex-patch-size-probs NAFLEX_PATCH_SIZE_PROBS [NAFLEX_PATCH_SIZE_PROBS ...]
                        Sampling probabilities for --naflex-patch-sizes.
  --naflex-seq-lens NAFLEX_SEQ_LENS [NAFLEX_SEQ_LENS ...]
                        Sequence lengths to sample for NaFlex training. Eval pads/crops
                        to the largest value.
  --naflex-seq-len-probs NAFLEX_SEQ_LEN_PROBS [NAFLEX_SEQ_LEN_PROBS ...]
                        Per-batch sampling weights for --naflex-seq-lens (same
                        length/order; normalized). Uniform if unset. NOTE: weights are
                        per BATCH; since B scales as budget/seq_len, a smaller seq-len
                        holds more rows, so the per-SAMPLE skew toward short seq-lens is
                        stronger than the weights suggest.
  --naflex-max-tokens-per-batch NAFLEX_MAX_TOKENS_PER_BATCH
                        Maximum tokens per local NaFlex batch. For GenLIP each row costs
                        image_seq_len + --naflex-max-text-tokens, so this is a total
                        (image+text) token budget; for image-only models it counts image
                        tokens.
  --naflex-max-text-tokens NAFLEX_MAX_TEXT_TOKENS
                        GenLIP caption token cap: truncates captions to this length AND
                        is added to the per-row token cost for batch sizing. Defaults to
                        the model's text_cfg.context_length when unset.
  --naflex-batch-divisor NAFLEX_BATCH_DIVISOR
                        Divisibility constraint for scheduled NaFlex batch sizes.
  --naflex-loss-scale {none,linear,sqrt}
                        Scale NaFlex training loss by actual local batch size relative
                        to --batch-size. Defaults to no scaling.
  --length-bucketing    Train: reorder by sequence length to reduce per-batch padding.
                        Standard CLIP / NaFlexClap key on caption (variable text);
                        NaFlexClap on audio; GenLAP on audio+caption; GenLIP on caption.
                        Reorder-only; train-only.
  --bucket-pool BUCKET_POOL
                        Per-worker sample pool size to sort for --length-bucketing
                        (bucketing breadth vs randomness). The pool buffers complete,
                        undecoded samples (raw image/audio bytes), so per-worker memory
                        scales with pool x average raw sample size.
  --bucket-chunk BUCKET_CHUNK
                        Run length within a sorted pool for --length-bucketing (~ a
                        typical batch size).
  --bucket-prefetch-pools BUCKET_PREFETCH_POOLS
                        Experimental: run --length-bucketing's pool fill (disk read +
                        tokenize) on a background thread, buffering this many flushed
                        pools ahead so it overlaps the batcher's decode instead of
                        alternating (smooths the disk/CPU/GPU seesaw). Buffers raw bytes
                        -> ~pool x this many extra samples in memory. 0 (default) keeps
                        the synchronous behavior.
  --naflex-pad-multiple NAFLEX_PAD_MULTIPLE
                        NaFlex audio only: pad to batch max, optionally rounded to
                        multiples of M and clamped at the per-batch cap. None = exact
                        batch-max. Use M (for example 32 or 64) to limit compile shapes.
  --text-pad-multiple TEXT_PAD_MULTIPLE
                        Variable-length text only: round per-batch caption length up to
                        multiples of M. None = exact batch-max. Use M (for example 16 or
                        32) to bound the number of distinct text sequence lengths and
                        limit torch.compile recompiles (the token-axis analogue of
                        --naflex-pad-multiple).
```

---

## Distillation (TinyCLIP)

Train a smaller student model from a larger pretrained teacher.

Example:

```bash
python -m open_clip_train.main \
    ... \
    --distill-model ViT-B-32 \
    --distill-pretrained openai
```

Replace the teacher model and checkpoint as required.

---

# Resume Training

```bash
python -m open_clip_train.main \
    --resume path/to/checkpoint.pt
```

---

# Evaluation

Run validation during training:

```bash
--val-data path/to/val.csv
```

Control evaluation frequency:

```bash
--val-frequency 1
```

Run zero-shot evaluation:

```bash
--zeroshot-frequency 1
```

---

# Checkpoints

Useful options:

```text
--save-frequency
--save-most-recent
--resume
```

---

# Logging

Supported logging backends:

* TensorBoard
* Weights & Biases
* Trackio

Examples:

```bash
--logs logs
--report-to tensorboard
```

---

# Precision

Supported precisions:

```text
amp
amp_bf16
bf16
fp16
fp32
```

Typical recommendation:

```bash
--precision amp
```

---

# Multi-GPU

Distributed training is supported using PyTorch Distributed.

Useful arguments:

```text
--dist-url
--dist-backend
--fsdp
```

---

# Common Useful Options

| Option             | Description               |
| ------------------ | ------------------------- |
| `--batch-size`     | Batch size per GPU        |
| `--epochs`         | Number of training epochs |
| `--workers`        | Data loader workers       |
| `--lr`             | Learning rate             |
| `--precision`      | Mixed precision mode      |
| `--model`          | Vision backbone           |
| `--pretrained`     | Initial weights           |
| `--resume`         | Resume training           |
| `--logs`           | Log directory             |
| `--save-frequency` | Checkpoint interval       |

---

# Full CLI Reference

To view every supported option:

```bash
python -m open_clip_train.main --help
```
