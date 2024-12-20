

python -u main_diffusion.py --gpus "0," --max_epochs=12  --num_workers=16 \
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
   --diffusion_block 2 \
   --steps 10  \
   --data d1125 \
   --kl_loss 1e-4 \
   --checkpoint your_checkpoint




