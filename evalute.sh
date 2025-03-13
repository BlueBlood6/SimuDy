python ./metrics.py \
    --num_imgs_per_class 2 \
    --tlr 1e-2 \
    --rlr 1e-2 \
    --bs 20 \
    --rbs 20 \
    --steps 8 \
    --linear 0 \
    --rec_lr 1e-2 \
    --eval_epoch 20 \
    --eval_step 10

python ./draw_pics.py \
    --num_imgs_per_class 2 \
    --tlr 1e-2 \
    --rlr 1e-2 \
    --bs 20 \
    --rbs 20 \
    --steps 8 \
    --linear 0 \
    --rec_lr 1e-2 \
    --eval_epoch 20 \
    --eval_step 10