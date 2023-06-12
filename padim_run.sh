#/bin/bash

function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

category=("Rivet_scr")
data_path=("/home/oshita/padim/datasets")
save_path=("./output_padim")
use_layers=("1-2-3")
arch=("resnet50")
Rd=("550")
non_Rd=("store_true")
seed=("64" "128" "256" "512" "1024")

for Rd in ${Rd[@]} ; do
    for layer in ${use_layer[@]} ; do
        for item in ${category[@]} ; do
            for arch in ${arch[@]} ; do
                for seed in ${seed[@]} ; do
                    for non_Rd in ${non_Rd[@]} ; do
                        for data_path in ${data_path[@]} ; do
                            for save_path in ${save_path[@]} ; do
                                python padim_main.py $Rd $layer $item $
                            done
                        done
                    done
                done
            done
        done
    done
done
