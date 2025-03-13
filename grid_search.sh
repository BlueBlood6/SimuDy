bs_of_recons=20
echo "Given batch size is $bs_of_recons"

for lr in $(seq 0.020 -0.001 0.001)
do
    python ./recons.py \
        --num_imgs_per_class 2 \
        --tlr 1e-2 \
        --bs 20 \
        --steps 8 \
        --rlr "$lr" \
        --rbs "$bs_of_recons" \
        --rec_lr 1e-2 \
        --grid_search 1
done