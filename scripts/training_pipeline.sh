#DATA_DIR="../benchmarks/msmarco-passage-ranking/data/processed"
#TRAIN_FILE="${DATA_DIR}/train/ms_train.jsonl"
#TRAIN_10K_FILE="${DATA_DIR}/train/ms_train_10k.jsonl"
#N=10000
#
#
#head -n $N $TRAIN_FILE > $TRAIN_10K_FILE

: "${USE_HF:=true}"
if [[ "$USE_HF" = true ]]; then
  echo "Training w/ HF: "
  python ../src/dense-retrieval-demo/hf_train.py
else
  echo "Training: "
  python ../src/dense-retrieval-demo/train.py
fi