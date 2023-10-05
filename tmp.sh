# Define arrays for the architectures, use_layers, and data_categories
# archs=("wide_resnet50_2" "swinv2_b")
# use_layers_list=("1" "2" "3" "4" "1-2" "1-3" "1-4" "2-3" "2-4" "3-4")

archs=("vit_b_16")
use_layers_list=("2" "5" "8" "11" "2-5" "2-8" "2-11" "5-8" "5-11" "8-11")
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
