CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --seed 14 \
    --encoder_name KMPNN \
    --batch_size 64 \
    --predictor_hidden_feats 32 \
    --patience 30 \
    --encoder_path /home/fangyin/kcl/dump/0910-Pretrain/gnn-kmpnn-model/KMPNNGNN_0910_2302_78000th_epoch.pkl \
    --readout_path /home/fangyin/kcl/dump/0910-Pretrain/gnn-kmpnn-model/Set2Set_0910_2302_78000th_epoch.pkl \
    --lr 0.01 \
    --predictor nonlinear \
    --eval nonfreeze \
    --data_name BACE \
    --split_type random \
    --dump_path ./dump \
    --exp_name KG-finetune-kmpnn \
    --exp_id bace-linear