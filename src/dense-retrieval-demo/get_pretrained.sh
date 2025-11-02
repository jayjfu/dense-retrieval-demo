mkdir data
cd data || exit

wget https://huggingface.co/google-bert/bert-base-uncased/resolve/main/pytorch_model.bin
wget https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json
wget https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt


cd - || exit