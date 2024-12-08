# Q1

python main.py --batch_size 32 --epochs 2
python main.py --batch_size 128 --epochs 2
python main.py --batch_size 512 --epochs 2
python main.py --batch_size 2048 --epochs 2

# Q2 & Q3 (only have access to two GPUs)

python main.py --batch_size 32 --epochs 2 --gpu_ids 0 1
python main.py --batch_size 128 --epochs 2 --gpu_ids 0 1
python main.py --batch_size 512 --epochs 2 --gpu_ids 0 1
python main.py --batch_size 2048 --epochs 2 --gpu_ids 0 1

# Q4

python main.py --batch_size 2048 --epochs 5 --gpu_ids 0 1