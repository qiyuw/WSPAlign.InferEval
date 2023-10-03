DATA_PATH=data # path to your data
mkdir $DATA_PATH

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