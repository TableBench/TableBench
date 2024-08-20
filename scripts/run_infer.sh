



DATA_PATH='data/v2/test'




MODEL_DIR='ckpt/llama-3-1-8b-v2'
python vllm_infer.py \
    --data_path $DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'tablebench'  \
    --temperature 0 \
    --sample_n 1 


