1. Please download the raw dataset to begin with: https://drive.google.com/drive/folders/1omNJYzx60kuTo9r5PteAK3OJM8SmEDpJ
Dataset orgranization:
<root_dir> -> <input_*> -> *.log 
2. Create .npy from the raw dataset file: python convert_logs_to_npy.py --input_root <input root dir> --output_root converted_npy --workers 4 --overwrite
3. Run the trainings: 
Unconditional Generation:
python train_<wgan/vae/diffusion>_pipeline_per_device_per_segment_update.py 
Conditional Generation:
python train_Masked_<wgan/vae/diffusion>_pipeline_per_device_per_segment_update.py 
4. Grep HD% values from the log files. 
5. Modify the paths accordingly in: result_plot.sh
6. Run: source result_plot.sh 
7. Get the plots in directory: hd_results 
