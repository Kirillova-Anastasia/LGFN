pip install tensorboard==1.11.0 tensorboardX==1.4
cd src/Deform_Conv
sh make.sh
cd ..
pip install matplotlib imageio scikit-image

python3 main.py --test_only --save_results --pre_train /workspace/LGFN/experiment/LGFN/LGFN_x4.pt --dir_demo /dataset --video_name test2_gauss_noise2 --save_dir /output
