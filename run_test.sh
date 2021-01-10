for i in 1111 2222 3333 4444 5555
do
    python -u main.py \
        --patience 10 \
        --stage2 5 \
        --nclasses 2 \
        --nhid 300 \
        --nlayers 1 \
        --attention-unit 350 \
        --word-vector "data/glove.42B.300d.txt.pt" \
        --traindev-data "data/train.csv" \
        --test-data "data/valid.csv" \
        --dictionary "data/vocab.json" \
        --lr 0.001 \
        --num-heads 8 \
        --batch-size 32 \
        --dropout 0.5 \
        --cuda \
        --seed $i
done
