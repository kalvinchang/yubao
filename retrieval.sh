#!/bin/bash

# speech-to-speech retrieval
langs=(standard_mandarin dialect_mandarin dialect_min dialect_wuu dialect_hak dialect_yue dialect_hsn dialect_gan)
gpu_count=8  # set to 1 if only 1 GPU
job_idx=0
pids=()

retrieval() {
    corpus=$1
    src=$2
    trg=$3
    gpu_id=$4
    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "[$(date)] Running $corpus $src -> $trg on GPU $gpu_id"

    (
        python retrieval.py ++exp_dir=MODEL_DIR \
            # where the results will be stored
            hydra.run.dir=retrieval/${corpus}_${src}_${trg} \
            ++inference.pretrained_model=PATH/TO/CHECKPOINT.pt \
            ++model.feats=fbank \
            ++data.test_data_config=retrieval_manifests.yaml \
            ++retrieval.source=${corpus}_${src} \
            ++retrieval.target=${corpus}_${trg}
    ) &

    wait
}

extract_recall() {
    exp_dir=$1
    corpus=$2
    src=$3
    trg=$4

    FILE=$(find "$exp_dir/retrieval" -type f -name "recall-${corpus}_${src}_${corpus}_${trg}*")

    if [[ -f "$FILE" ]]; then
        # last line in the file is the recall, averaged across all pairs of cities
        recall=$(tail -n 1 "$FILE" | awk '{printf "%.1f", $2}')
        echo -n " & $recall"
    else
        echo -n " & ???"
    fi
}

create_table() {
    exp_dir=$1

    echo '\begin{tabular}{l'"$(printf 'c%.0s' "${langs[@]}")"'}'
    echo '\toprule'

    # Column header
    echo -n "Source \textbackslash Target"
    for trg in "${langs[@]}"; do
        echo -n " & ${trg}"
    done
    echo " \\\\"
    echo '\midrule'

    # Table rows
    for src in "${langs[@]}"; do
        echo -n "$src"
        for trg in "${langs[@]}"; do
            if [ "$src" != "$trg" ]; then
                extract_recall "$exp_dir" "yubao" "$src" "$trg"
            else
                echo -n " & -"
            fi
        done
        echo " \\\\"
    done

    echo '\bottomrule'
    echo '\end{tabular}'
}

for src in "${langs[@]}"; do
    for trg in "${langs[@]}"; do
        if [ "$src" != "$trg" ]; then
            gpu_id=$(( job_idx % gpu_count ))
            retrieval "yubao" "$src" "$trg" "$gpu_id" &
            pids+=($!)
            ((job_idx++))

            # Wait if we've launched 8 full jobs
            if (( ${#pids[@]} == gpu_count )); then
                wait "${pids[@]}"
                pids=()
            fi
        fi
    done
done


# Final wait
wait "${pids[@]}"

# Creating a latex table
create_table MODEL_PATH
echo -e "\n\n\n\n"
