(python examples/cml_inter.py  swimmer_stan_1 --env_id SwimmerWrapped --seed 1;
python examples/cml_inter.py  swimmer_stan_2 --env_id SwimmerWrapped --seed 2;
python examples/cml_inter.py  swimmer_bf20_1 --env_id SwimmerWrapped --bf 20 --seed 1;
python examples/cml_inter.py  swimmer_bf20_2 --env_id SwimmerWrapped --bf 20 --seed 2
python examples/cml_inter.py  swimmer_bf50_1 --env_id SwimmerWrapped --bf 50 --seed 1;
python examples/cml_inter.py  swimmer_bf50_2 --env_id SwimmerWrapped --bf 50 --seed 2)&

(python examples/cml_inter.py swimmer_T50_1 --env_id SwimmerWrapped --T 50 --seed 1;
python examples/cml_inter.py swimmer_T50_2 --env_id SwimmerWrapped --T 50 --seed 2;
python examples/cml_inter.py swimmer_linear_1 --env_id SwimmerWrapped --linear 1 --seed 1;
python examples/cml_inter.py swimmer_linear_2 --env_id SwimmerWrapped --linear 1 --seed 2;
python examples/cml_inter.py swimmer_T50_1 --env_id SwimmerWrapped --T 50 --seed 1;
python examples/cml_inter.py swimmer_T50_2 --env_id SwimmerWrapped --T 50 --seed 2)
