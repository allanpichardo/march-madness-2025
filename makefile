.PHONY: train-fin-shooting train-fin-turnover train-fin-rebounding train-fin-defense train-fin-ft_foul train-fin-game_control pretrain train resume tensorboard predict-1 predict-2

train-fin-shooting:
	python train_fin.py --fin_key shooting --save_dir ./weights --epochs 5 --batch_size 1024

train-fin-turnover:
	python train_fin.py --fin_key turnover --save_dir ./weights --epochs 5 --batch_size 1024

train-fin-rebounding:
	python train_fin.py --fin_key rebounding --save_dir ./weights --epochs 5 --batch_size 1024

train-fin-defense:
	python train_fin.py --fin_key defense --save_dir ./weights --epochs 5 --batch_size 1024

train-fin-ft_foul:
	python train_fin.py --fin_key ft_foul --save_dir ./weights --epochs 5 --batch_size 1024

train-fin-game_control:
	python train_fin.py --fin_key game_control --save_dir ./weights --epochs 5 --batch_size 1024

pretrain: train-fin-shooting train-fin-turnover train-fin-rebounding train-fin-defense train-fin-ft_foul train-fin-game_control

train:
	python train_predictor.py --weights_dir ./weights --batch_size 128 --lr 1e-4 --epochs 50

resume:
	python train_predictor.py --weights_dir ./weights --batch_size 128 --lr 1e-4 --epochs 50 --resume

tensorboard:
	tensorboard --logdir=./logs --port=6006 --host=0.0.0.0

predict-1:
	python predict_brackets.py --csv_filename SampleSubmissionStage1.csv

predict-2:
	python predict_brackets.py --csv_filename SampleSubmissionStage2.csv