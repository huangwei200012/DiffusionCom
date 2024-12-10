

python main.py --gpus "0," --max_epochs=12  --num_workers=4 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --model_class UnimoKGC \
   --batch_size 96 \
   --label_smoothing 0.3 \
   --pretrain 0 \
   --bce 1 \
   --check_val_every_n_epoch 1 \
   --overwrite_cache \
   --data_dir dataset/FB15k-237 \
   --task_name fb15k-237 \
   --eval_batch_size 128 \
   --max_seq_length 64 \
   --lr 4e-5 \
   --data d1020 \
   --num_layers 2 \
   --hidden_size 1024 \
   --checkpoint your_checkpoint



