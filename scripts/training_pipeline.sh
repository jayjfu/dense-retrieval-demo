: "${USE_HF:=true}"
if [[ "$USE_HF" = true ]]; then
  echo "Training w/ HF: "
  python ../src/dense-retrieval-demo/hf_train.py --batch_size 128 --lr 5e-6 --num_epochs 2 --save_steps 20_000 --no_resume
else
  echo "Training: "
  python ../src/dense-retrieval-demo/train.py --batch_size 128 --lr 2e-5 --num_epochs 2 --logging_steps 2_000 --save_steps 20_000 --no_resume
fi