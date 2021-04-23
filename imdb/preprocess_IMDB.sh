wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'  

fairseq-preprocess \
    --only-source \
    --trainpref "aclImdb/train.input0.bpe" \
    --validpref "aclImdb/dev.input0.bpe" \
    --destdir "aclImdb/IMDB-bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "aclImdb/train.label" \
    --validpref "aclImdb/dev.label" \
    --destdir "aclImdb/IMDB-bin/label" \
    --workers 60 