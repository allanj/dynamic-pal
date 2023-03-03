# accelerate launch \
#     --config_file='/opt/tiger/trung_server/mwp/default_config.yaml' \
#     --num_process=8 \
#     --main_process_port='9873' \
#     scripts/prompt_codegen.py \
#         --batch_size 40 \
#         --lm_model_name math_corpus_v3_galactica-1.3b_seqlen_2048_global_step_95000 \
#         --base_model_dir /opt/tiger/trung_server/mwp \
#             2>&1 | tee prompt_results/math_corpus_v3_galactica-1.3b_seqlen_2048_global_step_95000.log


# accelerate launch \
#     --config_file='/opt/tiger/trung_server/mwp/default_config.yaml' \
#     --num_process=8 \
#     --main_process_port='9873' \
#     scripts/prompt_codegen.py \
#         --batch_size 40 \
#         --lm_model_name math_corpus_v3_galactica-1.3b_seqlen_2048_global_step_75000 \
#         --base_model_dir /opt/tiger/trung_server/mwp \
#             2>&1 | tee prompt_results/math_corpus_v3_galactica-1.3b_seqlen_2048_global_step_75000.log


# accelerate launch \
#     --config_file='/opt/tiger/trung_server/mwp/default_config.yaml' \
#     --num_process=8 \
#     --main_process_port='9873' \
#     scripts/prompt_codegen.py \
#         --batch_size 40 \
#         --lm_model_name math_corpus_v3_galactica-1.3b_seqlen_2048_global_step_75000 \
#         --base_model_dir /opt/tiger/trung_server/mwp \
#             2>&1 | tee prompt_results/math_corpus_v3_galactica-1.3b_seqlen_2048_global_step_75000.log



accelerate launch \
    --config_file='/opt/tiger/trung_server/mwp/default_config.yaml' \
    --num_process=8 \
    --main_process_port='9873' \
    scripts/prompt_codegen.py \
        --batch_size 40 \
        --lm_model_name galactica-1.3b \
        --base_model_dir /mnt/bn/trung-nas/hf_models \
        2>&1 | tee prompt_results/original.log



