export OPENAI_BASE_URL=http://localhost:8877/v1
python run_blocksworld_batch_llm.py \
    --instance-start 1 --instance-end 100 \
    --output-json results/llm_batch_generated_basic_1_100.json \
    --llm-model Qwen/Qwen3-32B-FP8 \
    --llm-response-format json_object \
    --quiet