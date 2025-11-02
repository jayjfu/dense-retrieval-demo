# MS MARCO passage ranking benchmark
task_dir='../benchmarks/msmarco-passage-ranking/'

# Reference file
reference_file="$task_dir/data/qrels.dev.tsv"

# Prediction file
candidate_file="$task_dir/eval/hf_bert.ranking_results.dev.tsv"

# Run MS MARCO official evaluation script
python ../benchmarks/msmarco-passage-ranking/eval/ms_marco_eval.py $reference_file $candidate_file