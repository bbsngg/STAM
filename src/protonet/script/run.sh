
python -u ../train.py \
--cuda-id=0 \
--task-name=huff \
--loss-method=SimCLR+KL \
--n-way=5 --k-shot=5 --seed=42 --num-epochs=2 \
--train-epi=2000 --dev-epi=500 --test-epi=1800 --eval-step=3000 \
--aug-method=mix  \
--contrast-weight=0.01 \
--kl-weight=0.1 \
--save-model \
> fsl_log/huff.log

