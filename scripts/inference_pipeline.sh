PROJECT_ROOT="../src/dense-retrieval-demo"
cd $PROJECT_ROOT || exit

# Indexing
python -m inference.encoding

# FAISS Search
python -m inference.vector_search

cd - || exit