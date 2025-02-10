# base run
source ~/miniconda3/bin/activate myenv
# lr experiment
# at present, i'm not focusing on the asymptotic behavior of these runs
# since this is just to gauge which hyperparameters are *significantly* different
# which should be noticeable without million episode runs
# disabled these runs because saved .npy and plot already
# python main_pre.py -c config_mp.ini --max_eps=1_400_000 --n_run=3 --name="3 layers"
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.000004]' --max_eps=1_400_000 --n_run=3 --name="4e-6"
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.0000004]' --max_eps=1_400_000 --n_run=3 --name="4e-7"
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00000004]' --max_eps=1_400_000 --n_run=3 --name="4e-8"
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00004]' --max_eps=1_400_000 --n_run=3 --name="lr=4e-5" --exp_num=3
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.0004]' --max_eps=1_400_000 --n_run=3 --name="lr=4e-4"
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.004]' --max_eps=1_400_000 --n_run=3 --name="lr=4e-3"
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.04]' --max_eps=1_400_000 --n_run=3 --name="lr=4e-2"
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.4]' --max_eps=1_400_000 --n_run=3 --name="lr=4e-1"
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00001]' --max_eps=1_400_000 --n_run=3 --name="lr=1e-5" --exp_num=3
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00002]' --max_eps=1_400_000 --n_run=3 --name="lr=2e-5" --exp_num=3
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00003]' --max_eps=1_400_000 --n_run=3 --name="lr=3e-5" --exp_num=3
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00004]' --max_eps=1_400_000 --n_run=3 --name="lr=5e-5" --exp_num=3
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="lr=6e-5" --exp_num=3
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00007]' --max_eps=1_400_000 --n_run=3 --name="lr=7e-5" --exp_num=3
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00008]' --max_eps=1_400_000 --n_run=3 --name="lr=8e-5" --exp_num=3
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00009]' --max_eps=1_400_000 --n_run=3 --name="lr=9e-5" --exp_num=3
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.000001]' --max_eps=1_400_000 --n_run=3 --name="lr=1e-6" --exp_num=4
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.000002]' --max_eps=1_400_000 --n_run=3 --name="lr=2e-6" --exp_num=4
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.000003]' --max_eps=1_400_000 --n_run=3 --name="lr=3e-6" --exp_num=4
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.000005]' --max_eps=1_400_000 --n_run=3 --name="lr=5e-6" --exp_num=4
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.000006]' --max_eps=1_400_000 --n_run=3 --name="lr=6e-6" --exp_num=4
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.000007]' --max_eps=1_400_000 --n_run=3 --name="lr=7e-6" --exp_num=4
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.000008]' --max_eps=1_400_000 --n_run=3 --name="lr=8e-6" --exp_num=4
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.000009]' --max_eps=1_400_000 --n_run=3 --name="lr=9e-6" --exp_num=4

# lr = 6e-5
# second layer var
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.1" --exp_num=5
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.2,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.2" --exp_num=5
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.3,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.3" --exp_num=5
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.4,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.4" --exp_num=5
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.5,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.5" --exp_num=5
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.6" --exp_num=5
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.7,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.7" --exp_num=5
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.8,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.8" --exp_num=5
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.9,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.9" --exp_num=5
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,1,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=1.0" --exp_num=5

# third layer var
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.1" --exp_num=6
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.2,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.2" --exp_num=6
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.3,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.3" --exp_num=6
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.4,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.4" --exp_num=6
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.5,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.5" --exp_num=6
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.6,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.6" --exp_num=6
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.7,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.7" --exp_num=6
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.8" --exp_num=6
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.9,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.9" --exp_num=6
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,1,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=1" --exp_num=6

# last layer var
# avg. 0.76 median 0.77 min 0.74 max 0.78 std 0.01
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.1" --exp_num=7
# # avg. 0.75 median 0.74 min 0.74 max 0.75 std 0.00
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.2]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.2" --exp_num=7
# # avg. 0.76 median 0.75 min 0.75 max 0.77 std 0.01
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.3]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.3" --exp_num=7
# # avg. 0.76 median 0.76 min 0.75 max 0.77 std 0.01
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.4]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.4" --exp_num=7
# # avg. 0.75 median 0.74 min 0.74 max 0.76 std 0.01
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.5]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.5" --exp_num=7
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.6]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.6" --exp_num=7
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.7]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.7" --exp_num=7
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.8]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.8" --exp_num=7
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.9]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=0.9" --exp_num=7
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="var=1" --exp_num=7

# test with 32 units
# python main_pre.py -c config_mp.ini --hidden='[64,32,32]' --var='[0.3,0.6,0.8,0.1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="lr=6e-5" --exp_num=8
# python main_pre.py -c config_mp.ini --hidden='[64,32,32]' --var='[0.3,0.6,0.8,0.1]' --lr='[0.04,0.00004,0.000004, 0.000006]' --max_eps=1_400_000 --n_run=3 --name="lr=6e-6" --exp_num=8
# python main_pre.py -c config_mp.ini --hidden='[64,32,32]' --var='[0.3,0.6,0.8,0.1]' --lr='[0.04,0.00004,0.000004, 0.0006]' --max_eps=1_400_000 --n_run=3 --name="lr=6e-4" --exp_num=8
# python main_pre.py -c config_mp.ini --hidden='[64,32,32]' --var='[0.3,0.6,0.8,0.1]' --lr='[0.04,0.00004,0.000004, 0.006]' --max_eps=1_400_000 --n_run=3 --name="lr=6e-3" --exp_num=8
# python main_pre.py -c config_mp.ini --hidden='[64,32,32]' --var='[0.3,0.6,0.8,0.1]' --lr='[0.04,0.00004,0.000004, 0.000006]' --max_eps=1_400_000 --n_run=3 --name="lr=6e-7" --exp_num=8
# python main_pre.py -c config_mp.ini --max_eps=1_400_000 --n_run=3 --name="3 layers" --exp_num=8
# python main_pre.py -c config_mp.ini --hidden='[64,32,16]' --var='[0.3,0.6,0.8,0.1]' --lr='[0.04,0.00004,0.000004, 0.00006]' --max_eps=1_400_000 --n_run=3 --name="16 units" --exp_num=8
# last layer 32 units
python test.py --exp_num=8
conda deactivate