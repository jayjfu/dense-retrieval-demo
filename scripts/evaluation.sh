# MS MARCO passage ranking benchmark
TASK_DIR="../benchmarks/msmarco-passage-ranking/"

# Reference file
REFERENCE_FILE="$TASK_DIR/data/qrels.dev.tsv"

# Prediction file
CANDIDATE_FILE="$TASK_DIR/eval/hf_bert.ranking_results.dev.tsv"

# Run MS MARCO official evaluation script
python ../benchmarks/msmarco-passage-ranking/eval/ms_marco_eval.py $REFERENCE_FILE $CANDIDATE_FILE