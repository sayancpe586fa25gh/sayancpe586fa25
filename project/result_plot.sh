python result_plot.py --log run_log_vae/test_5_hd.log --op hd_results/vae_results --key VAE
python result_plot.py --log run_log_vae/masked_test_3_hd.log --op hd_results/conditional_vae_results --key "Conditional VAE"
python result_plot.py --log run_log_wgan/test_1_hd.log --op hd_results/wgan_results --key WGAN
python result_plot.py --log run_log_wgan/masked_test_1_hd.log --op hd_results/conditional_wgan_results --key "Conditional WGAN"
python result_plot.py --log run_log_diffusion/test_1_hd.log --op hd_results/diffusion_results --key Diffusion
python result_plot.py --log run_log_diffusion/Masked_train_1_hd.log --op hd_results/conditional_diffusion_results --key "Conditional Diffusion"
