# Define arrays for the architectures, use_layers, and data_categories
archs=("conformer_b_16_cnn" "conformer_b_16_vit")
use_layers_list=("3" "7" "10" "11" "3-11" "3-7" "3-10" "7-10" "7-11" "10-11")
data_categories=("hazelnut" "cable" "carpet" "bottle" "capsule" "grid" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "zipper" "wood")

# Loop through the combinations
for arch in "${archs[@]}"; do
    for use_layers in "${use_layers_list[@]}"; do
        for data_category in "${data_categories[@]}"; do
            for seed in {0..2}; do
                python padim_main.py --seed "$seed" --data_category "$data_category" --arch "$arch" --use_layers "$use_layers"
            done
        done
    done
done
