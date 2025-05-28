pip install 'numpy<2.0.0' packaging
pip install --upgrade setuptools
pip install deepspeed==0.12.6 --prefer-binary
pip install colorlog
pip install einops
pip install lpips
pip install scikit-video
cd /mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan
pip install -r requirements.txt
pip install --upgrade diffusers
sudo apt update
sudo apt install libgl1-mesa-glx -y
sudo apt install net-tools
# wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth -P ~/.cache/torch/hub/checkpoints/
# wget "https://download.pytorch.org/models/vgg16-397923af.pth" -P ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
cd /mnt/bn/occupancy3d/workspace
sudo chown -R tiger:tiger mzj
chmod -R u+rwx mzj
cd /mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan
mkdir -p ~/.cache/torch/hub/checkpoints/
cp ./alexnet-owt-7be5be79.pth ~/.cache/torch/hub/checkpoints/
cp ./vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/
echo "all is ok!"


