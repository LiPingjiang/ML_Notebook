import indw2v

embedder = indw2v.INDW2V(INDEX_SIZE=8,EMBED_DIM=100,CONTEXT_SIZE=4)
embedder.load('data/index_input')
embedder.trian()