docker run -it -v ~/mnt/calypso/24a_kir/MSU_VSR_Benchmark/input:/dataset:ro -v ~/mnt/calypso/24a_kir/MUS_VSR_Benchmark/result_on_noise2 --gpus '"device=2"' pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel
pip install tensorboard==1.11.0 tensorboardX==1.4
git clone https://github.com/Kirillova-Anastasia/LGFN.git
cd LGFN/src/Deform_Conv
sh make.sh
cd ..
pip install matplotlib imageio scikit-image
python3 main.py --test_only --save_results --pre_train /workspace/LGFN/experiment/LGFN/LGFN_x4.pt --dir_demo /dataset/test1_bicubic_noise2 --dir_demo_GT /dataset/test1_bicubic_noise2 --save_dir /output/test1_bicubic_noise2
