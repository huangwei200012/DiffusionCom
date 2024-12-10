# Required environment variables:
# model_name_or_path: pre-trained text model name ( bert-*)
# checkpoint: the path of the pretrained model
# batch_size: batch size (recommendation: 96)
# lr: learning rate (recommendation: 4e-5)

python -u main.py --gpus "0," --max_epochs=12  --num_workers=4 \
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
   --checkpoint /root/autodl-tmp/MKGformer-main/MKG/output/FB15k-237/epoch=11-Eval/hits10=0.50-Eval/hits1=0.26.ckpt
   # > /root/autodl-tmp/MKGformer-main/MKG/log/train_log_pre0


