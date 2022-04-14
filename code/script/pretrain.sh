CUDA_VISIBLE_DEVICES=0 python pretrain.py \
    --hidden_feats 64 \
    --num_gnn_layers 2 \
    --epoch_num 100 \
    --data_name  ../data/cluster_0.85 \
    --lmdb_env ../data/zinc15_250K_2D \
    --generator_process before_encoder \
    --dump_path ./dump \
    --exp_name Pretrain \
    --exp_id gnn-kmpnn
    # --dropout 0.1 \
