python gen_model_impl.py --model vae --epochs 1000 --save_every 200 --data_fraction 0.03 > gen_model_run_vae.res
python gen_model_impl.py --model gan --epochs 1000 --save_every 200 --data_fraction 0.03 > gen_model_run_gan.res
python gen_model_impl.py --model diffusion --epochs 1000 --save_every 200 --data_fraction 0.03 > gen_model_run_diff.res
