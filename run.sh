# restaurant
python train.py --model_name td_bert --dataset restaurant                                 --adv 0
python train.py --model_name td_bert --dataset restaurant_random_rest_test                --adv 0
python train.py --model_name td_bert --dataset restaurant_random_laptop_test              --adv 0
python train.py --model_name td_bert --dataset restaurant_random_rest_train_gold_test     --adv 0
python train.py --model_name td_bert --dataset restaurant_random_laptop_train_gold_test   --adv 0

python train.py --model_name td_bert --dataset restaurant                                 --adv 0.1
python train.py --model_name td_bert --dataset restaurant_random_rest_test                --adv 0.1
python train.py --model_name td_bert --dataset restaurant_random_laptop_test              --adv 0.1
python train.py --model_name td_bert --dataset restaurant_random_rest_train_gold_test     --adv 0.1
python train.py --model_name td_bert --dataset restaurant_random_laptop_train_gold_test   --adv 0.1

python train.py --model_name td_bert --dataset restaurant                                 --adv 0.5
python train.py --model_name td_bert --dataset restaurant_random_rest_test                --adv 0.5
python train.py --model_name td_bert --dataset restaurant_random_laptop_test              --adv 0.5
python train.py --model_name td_bert --dataset restaurant_random_rest_train_gold_test     --adv 0.5
python train.py --model_name td_bert --dataset restaurant_random_laptop_train_gold_test   --adv 0.5


# laptop
python train.py --model_name td_bert --dataset laptop                                 --adv 0
python train.py --model_name td_bert --dataset laptop_random_laptop_test              --adv 0
python train.py --model_name td_bert --dataset laptop_random_rest_test                --adv 0
python train.py --model_name td_bert --dataset laptop_random_rest_train_gold_test     --adv 0
python train.py --model_name td_bert --dataset laptop_random_laptop_train_gold_test   --adv 0

python train.py --model_name td_bert --dataset laptop                                 --adv 0.1
python train.py --model_name td_bert --dataset laptop_random_laptop_test              --adv 0.1
python train.py --model_name td_bert --dataset laptop_random_rest_test                --adv 0.1
python train.py --model_name td_bert --dataset laptop_random_rest_train_gold_test     --adv 0.1
python train.py --model_name td_bert --dataset laptop_random_laptop_train_gold_test   --adv 0.1

python train.py --model_name td_bert --dataset laptop                                 --adv 0.5
python train.py --model_name td_bert --dataset laptop_random_laptop_test              --adv 0.5
python train.py --model_name td_bert --dataset laptop_random_rest_test                --adv 0.5
python train.py --model_name td_bert --dataset laptop_random_rest_train_gold_test     --adv 0.5
python train.py --model_name td_bert --dataset laptop_random_laptop_train_gold_test   --adv 0.5





#   dataset: ['twitter', 'restaurant', 'restaurant_random_rest_test', 'restaurant_random_laptop_test', 'restaurant_random_rest_train_gold_test', 'restaurant_random_laptop_train_gold_test', 'laptop', 'laptop_random_laptop_test', 'laptop_random_rest_test', 'laptop_random_rest_train_gold_test', 'laptop_random_laptop_train_gold_test']