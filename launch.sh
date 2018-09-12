nohup python ./train.py --gpu_id=0 --val_num=1 1>log1 2>log1 &
nohup python ./train.py --gpu_id=0 --val_num=2 1>log2 2>log2 &
nohup python ./train.py --gpu_id=1 --val_num=3 1>log3 2>log3 &
nohup python ./train.py --gpu_id=1 --val_num=4 1>log4 2>log4 &
nohup python ./train.py --gpu_id=2 --val_num=5 1>log5 2>log5 &
nohup python ./train.py --gpu_id=2 --val_num=6 1>log6 2>log6 &
nohup python ./train.py --gpu_id=2 --val_num=7 1>log7 2>log7 &
