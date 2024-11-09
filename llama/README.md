# Use Llama-2-7b-chat to generate SysCaps

## Dependencies

You must have access to the Llama-2-7b-chat model weights. The model is not included in this repository. You can download the model weights from the Hugging Face model hub: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf or from Meta (`consolidated.00.pth`). 

Install the following dependencies in your environment:

- xt
- torch
- fairscale
- fire
- sentencepiece

### ComStock

```
torchrun --nproc_per_node 1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    generate_building_syscaps.py \
    --ckpt_dir $PATH_TO_MODEL_WEIGHTS/llama-2-7b-chat/ \
    --tokenizer_path $PATH_TO_MODEL_WEIGHTS/tokenizer.model \
    --max_seq_len 512 --max_batch_size 4 \
    --prompt_length all
```

### Wind

```
torchrun --nproc_per_node 1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    generate_wind_syscaps.py \
    --ckpt_dir $PATH_TO_MODEL_WEIGHTS//llama-2-7b-chat/ \
    --tokenizer_path $PATH_TO_MODEL_WEIGHTS//tokenizer.model \
    --max_seq_len 512 --max_batch_size 4 \
    --prompt_augmentation all
```

### Sampling building type synonyms

```
torchrun --nproc_per_node 1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    sample_building_type_synonyms.py \
    --ckpt_dir $PATH_TO_MODEL_WEIGHTS//llama-2-7b-chat/ \
    --tokenizer_path $PATH_TO_MODEL_WEIGHTS//tokenizer.model \
    --max_seq_len 512 --max_batch_size 4 
```