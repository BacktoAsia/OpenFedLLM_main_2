# 交互式命令，适合debug，因为Max Walltime只有8小时
srun -p interactive -J test --gres=gpu:1 --mem-per-cpu=16G --cpus-per-task 2 --time 08:00:00 --pty bash -i

