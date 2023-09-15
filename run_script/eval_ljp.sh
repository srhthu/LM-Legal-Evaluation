CUDA_VISIBLE_DEVICES=1 python eval_pipeline.py \
`# python -m llm_eval.task.ljp` \
./outputs/ljp/baichuan_7b \
--sub_tasks all \
--agent_type hf \
--model /storage_fast/rhshui/llm/baichuan-7b --trust_remote_code true \
--num_output 5 \
--max_new_tokens 30 \
--do_sample \
--temperature 0.8 \
--prompt_config_file /storage/rhshui/workspace/LJP/lm/runs/benchmark/config.json \
--meta_data_path /storage/rhshui/workspace/LJP/lm/runs/benchmark/meta_data.json \
--train_data_path /storage/rhshui/workspace/LJP/lm/data/sample_train_n10_seed2222.json \
--test_data_path /storage/rhshui/workspace/LJP/lm/data/sample_test_n5_seed2222.json \
--label_path /storage/rhshui/workspace/LJP/dataset/cail_clean/charge2id.json \
--task_type ljp \