python train.py --model_name bert_spc --dataset restaurant               --max_seq_len 128 --num_epoch 20 --adv 0.5
python train.py --model_name bert_spc --dataset restaurant_random_laptop --max_seq_len 128 --num_epoch 20 --adv 0.5
python train.py --model_name bert_spc --dataset restaurant_random_rest   --max_seq_len 128 --num_epoch 20 --adv 0.5
python train.py --model_name bert_spc --dataset restaurant               --max_seq_len 128 --num_epoch 20 --adv 0.1
python train.py --model_name bert_spc --dataset restaurant_random_laptop --max_seq_len 128 --num_epoch 20 --adv 0.1
python train.py --model_name bert_spc --dataset restaurant_random_rest   --max_seq_len 128 --num_epoch 20 --adv 0.1