DATA_PATH=data # path to your data
mkdir $DATA_PATH

# test dataset
mkdir $DATA_PATH/test_data
for LANG in kftt deen enfr roen
do
    wget https://huggingface.co/datasets/qiyuw/wspalign_test_data/blob/main/"$LANG"_test.json -O "$DATA_PATH"/few_ft_data/"$LANG"_test.json
done

# ground truth
mkdir $DATA_PATH/wspalign_acl2023_eval
for LANG in deen enfr roen
do
    mkdir $DATA_PATH/wspalign_acl2023_eval/"$LANG"
    for MODE in moses text json
    do
        wget https://huggingface.co/datasets/qiyuw/wspalign_acl2023_eval/resolve/main/"$LANG"/"$LANG"_test."$MODE" -O "$DATA_PATH"/wspalign_acl2023_eval/"$LANG"/"$LANG"_test."$MODE"
    done
done

# special handling for kftt
mkdir $DATA_PATH/wspalign_acl2023_eval/kftt
for MODE in moses txt json
do
    wget https://huggingface.co/datasets/qiyuw/wspalign_acl2023_eval/resolve/main/kftt/kftt_devtest."$MODE" -O "$DATA_PATH"/wspalign_acl2023_eval/kftt/kftt_devtest."$MODE"
done