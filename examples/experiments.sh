(python examples/cml_inter.py  swimmer_stan_1 --env_id SwimmerWrapped --seed 1 --gpu_id 1 &> swimmer_stan_1.txt;
python examples/cml_inter.py  swimmer_stan_2 --env_id SwimmerWrapped --seed 2 --gpu_id 1 &> swimmer_stan_2.txt;
python examples/cml_inter.py  swimmer_bf20_1 --env_id SwimmerWrapped --bf 20 --seed 1 --gpu_id 1 &> swimmer_bf20_1.txt;
python examples/cml_inter.py  swimmer_bf20_2 --env_id SwimmerWrapped --bf 20 --seed 2 --gpu_id 1 &> swimmer_bf20_2.txt
python examples/cml_inter.py  swimmer_bf50_1 --env_id SwimmerWrapped --bf 50 --seed 1 --gpu_id 1 &> swimmer_bf50_1.txt;
python examples/cml_inter.py  swimmer_bf50_2 --env_id SwimmerWrapped --bf 50 --seed 2 --gpu_id 1 &> swimmer_bf50_2.txt)&

(python examples/cml_inter.py swimmer_T50_1 --env_id SwimmerWrapped --T 50 --seed 1 --gpu_id 5 &> swimmer_T50_1.txt;
python examples/cml_inter.py swimmer_T50_2 --env_id SwimmerWrapped --T 50 --seed 2 --gpu_id 5 &> swimmer_T50_2.txt;
python examples/cml_inter.py swimmer_linear_1 --env_id SwimmerWrapped --linear 1 --seed 1 --gpu_id 5 &> swimmer_linear_1.txt;
python examples/cml_inter.py swimmer_linear_2 --env_id SwimmerWrapped --linear 1 --seed 2 --gpu_id 5 &> swimmer_linear_2.txt;
python examples/cml_inter.py swimmer_maxiter80 --env_id SwimmerWrapped --maxiter 80 --seed 1 --gpu_id 5 &> swimmer_iter80_1.txt;
python examples/cml_inter.py swimmer_maxiter80 --env_id SwimmerWrapped --mxiter 80 --seed 2 --gpu_id 5 &> swimmer_iter80_2.txt)
