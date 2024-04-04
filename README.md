Generator
https://github.com/mistralai/mistral-src/tree/main

Retriever 
https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

https://github.com/microsoft/LoRA

https://shamanesiri.medium.com/how-to-finetune-the-entire-rag-architecture-including-dpr-retriever-4b4385322552

https://andrew.gibiansky.com/facebooks-knowledge-assisted-nlp/

https://github.com/facebookresearch/faiss/issues/2078

https://docs.mistral.ai/guides/basic-RAG/

https://huggingface.co/NousResearch/Genstruct-7B

https://web.archive.org/web/20230215061030/https://the-eye.eu/public/Books/ManyThings/DOC/

Generator
https://github.com/mistralai/mistral-src/tree/main

Retriever 
https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

https://github.com/microsoft/LoRA

https://shamanesiri.medium.com/how-to-finetune-the-entire-rag-architecture-including-dpr-retriever-4b4385322552

https://andrew.gibiansky.com/facebooks-knowledge-assisted-nlp/

https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning#io-and-deep-copying-indexes
https://github.com/facebookresearch/faiss/issues/2078

https://docs.mistral.ai/guides/basic-RAG/

https://github.com/david-smejkal/wiki2txt
# rag

# RUN
build index on data

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 --master_port $RANDOM -m train
wget -c https://models.mistralcdn.com/mistral-7b-v0-2/Mistral-7B-v0.2-Instruct.tar
tar -xf Mistral-7B-v0.2-Instruct.tar

File "/workspace/rag/finetune/lora/linear.py", line 113, in _load_from_state_dict
    assert self.quantized, "Provided checkpoint is already quantized"
AssertionError: Provided checkpoint is already quantized
