import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
import argparse
import logging
from colorlog import ColoredFormatter
import tqdm
from itertools import chain
import wandb
import random
import numpy as np
from pathlib import Path
from einops import rearrange
import time
from utils import visualize_input_and_recon, save_video_comparison, log_fixed_val_video
try:
    import lpips
except:
    raise Exception("Need lpips to valid.")
import sys
sys.path.append(".")
from opensora.models.causalvideovae.model import * # 导入 model 目录下的 __init__.py 文件
from opensora.models.causalvideovae.model.ema_model import EMA
from opensora.models.causalvideovae.dataset.ddp_sampler import CustomDistributedSampler
from opensora.models.causalvideovae.dataset.video_dataset import TrainVideoDataset, ValidVideoDataset
from opensora.models.causalvideovae.dataset.fakevideo_dataset import MultiViewSequenceDataset
from opensora.models.causalvideovae.model.utils.module_utils import resolve_str_to_obj
from opensora.models.causalvideovae.utils.video_utils import tensor_to_video


def safe_bar_update(bar, n=1):
    try:
        if bar is not None:
            bar.update(n)
    except Exception as e:
        print(f"[WARNING] bar.update() failed: {e}")

def set_random_seed(seed):
    # 用于设置 Python、NumPy 和 PyTorch 的随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def setup_logger(rank): #??
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        f"[rank{rank}] %(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        reset=True,
        style="%",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(stream_handler)

    return logger

def check_unused_params(model): # 检查未参与反向传播的参数
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    return unused_params

def set_requires_grad_optimizer(optimizer, requires_grad): # 控制参数是否参与梯度计算 #??
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param.requires_grad = requires_grad

def total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1e6
    return int(total_params_in_millions)


def get_exp_name(args):  # 根据训练参数生成实验名称
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # return f"{args.exp_name}-{timestamp}-lr{args.lr:.2e}-bs{args.batch_size}-rs{args.resolution}-sr{args.sample_rate}-fr{args.num_frames}"
    return f"{args.exp_name}-{timestamp}"


def set_train(modules):
    for module in modules:
        module.train()


def set_eval(modules):
    for module in modules:
        module.eval()


def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)


def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict={},
):
    filepath = checkpoint_dir / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
        },
        filepath,
    )
    return filepath


def valid(global_rank, rank, model, val_dataloader, precision, args, epoch=None, step=None):
    if args.eval_lpips:
        lpips_model = lpips.LPIPS(net="alex", spatial=True) # spatial=True 表示输出每个像素的 LPIPS 值（后续求平均）
        lpips_model.to(rank)
        lpips_model = DDP(lpips_model, device_ids=[rank])
        lpips_model.requires_grad_(False)
        lpips_model.eval()

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_list = [] # 每个 batch 的 PSNR 值（衡量图像质量）
    lpips_list = []
    video_log = []
    num_video_log = args.eval_num_video_log

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["video"].to(rank) # # [B, C, T, H, W]
            with torch.cuda.amp.autocast(dtype=precision):
                outputs = model(inputs)
                video_recon = outputs[0]

            # Upload videos
            if global_rank == 0:
                for i in range(len(video_recon)):
                    if num_video_log <= 0:
                        break
                    video = tensor_to_video(video_recon[i])
                    video_log.append(video)
                    num_video_log -= 1
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            video_recon = rearrange(
                video_recon, "b c t h w -> (b t) c h w"
            ).contiguous()

            # Calculate PSNR
            mse = torch.mean(torch.square(inputs - video_recon), dim=(1, 2, 3))
            psnr = 20 * torch.log10(1 / torch.sqrt(mse))
            psnr = psnr.mean().detach().cpu().item()

            # Calculate LPIPS
            if args.eval_lpips:
                lpips_score = (
                    lpips_model.forward(inputs, video_recon)
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
                lpips_list.append(lpips_score)

            psnr_list.append(psnr)
            if global_rank == 0:
                bar.update()
            # Release gpus memory
            torch.cuda.empty_cache()

    logging.info(f"video_log: {len(video_log)}")
    logging.info(f"psnr_list: {len(psnr_list)}")
    logging.info(f"lpips_list: {len(lpips_list)}")
    logging.info(f"Saving {num_video_log} samples")

    return psnr_list, lpips_list, video_log


def gather_valid_result(psnr_list, lpips_list, video_log_list, rank, world_size):#?
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_video_logs = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_video_logs, video_log_list)
    return (
        np.array(gathered_psnr_list).mean(),
        np.array(gathered_lpips_list).mean(),
        list(chain(*gathered_video_logs)),
    )


def train(args):
    # Setup logger
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    logger = setup_logger(rank)

    # 创建检查点保存文件夹
    ckpt_dir = Path(args.ckpt_dir) / Path(get_exp_name(args))
    if global_rank == 0:
        try:
            ckpt_dir.mkdir(exist_ok=False, parents=True) # parents=True 表示如果上级目录不存在，也会一并创建
        except:
            logger.warning(f"`{ckpt_dir}` exists!")
            time.sleep(5)
    dist.barrier()

    # Load generator model，任何时候 print(model.config) 就知道模型是怎么建的，可追溯、可复现
    model_cls = ModelRegistry.get_model(args.model_name)
    if not model_cls:
        raise ModuleNotFoundError(
            f"`{args.model_name}` not in {str(ModelRegistry._models.keys())}."
        )
    
    if args.pretrained_model_name_or_path is not None:
        if global_rank == 0:
            logger.warning(
                f"You are loading a checkpoint from `{args.pretrained_model_name_or_path}`."
            )
        model = model_cls.from_pretrained( # VideoBaseAE中和Mixin中都定义
            args.pretrained_model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    else:
        if global_rank == 0:
            logger.warning(f"Model will be inited randomly.")
        model = model_cls.from_config(args.model_config)
    
    if global_rank == 0: #wandb #???
        logger.warning("Connecting to WANDB...")
        model_config = dict(**model.config)
        args_config = dict(**vars(args))
        if 'resolution' in model_config:
            del model_config['resolution']
        
        wandb.init(
            mode="offline",
            project=os.environ.get("WANDB_PROJECT", "causalvideovae"),
            config=dict(**model_config, **args_config),
            name=get_exp_name(args),
        )
    
    dist.barrier()
    
    # Load discriminator model
    disc_cls = resolve_str_to_obj(args.disc_cls, append=False)
    logger.warning(
        f"disc_class: {args.disc_cls} perceptual_weight: {args.perceptual_weight}  loss_type: {args.loss_type}"
    )
    disc = disc_cls(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        kl_weight=args.kl_weight,
        logvar_init=args.logvar_init,
        perceptual_weight=args.perceptual_weight,
        loss_type=args.loss_type,
        wavelet_weight=args.wavelet_weight
    )

    # DDP
    model = model.to(rank, )
    model = DDP(
        model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )
    disc = disc.to(rank)
    disc = DDP(
        disc, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )

    # Load dataset
    # dataset = TrainVideoDataset(
    #     args.video_path,
    #     sequence_length=args.num_frames,
    #     resolution=args.resolution,
    #     sample_rate=args.sample_rate,
    #     dynamic_sample=args.dynamic_sample,
    #     cache_file="idx.pkl",
    #     is_main_process=global_rank == 0,
    # )
    dataset = MultiViewSequenceDataset(
        pkl_path="/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/pkls/dcr_data_dynamic_bottom_scaleddepth.pkl",
        # pkl_path="/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/pkls/dcr_lf_sub.pkl",
        resolution=args.resolution,
        sequence_length=3
    )
    ddp_sampler = CustomDistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=ddp_sampler,
        pin_memory=True, # PyTorch 会将数据预加载到 CUDA 的锁页内存中
        num_workers=args.dataset_num_worker,
    )
    print(f"[INFO] Training set loaded: {len(dataset)} videos.")
    # val_dataset = ValidVideoDataset(
    #     real_video_dir=args.eval_video_path,
    #     num_frames=args.eval_num_frames,
    #     sample_rate=args.eval_sample_rate,
    #     crop_size=args.eval_resolution,
    #     resolution=args.eval_resolution,
    # )
    val_dataset = MultiViewSequenceDataset(
        pkl_path="/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/pkls/dcr_data_dynamic_bottom_scaleddepth.pkl",
        # pkl_path="/mnt/bn/occupancy3d/workspace/lzy/robotics-data-sdk/data_infos/pkls/dcr_lf_sub.pkl",
        # resolution=512,
        resolution=args.resolution,
        sequence_length=3
    )
    print(f"[INFO] Validation set loaded: {len(val_dataset)} videos.")
    subset_size = min(args.eval_subset_size, len(val_dataset))
    indices = range(subset_size)
    val_dataset = Subset(val_dataset, indices=indices)
    val_sampler = CustomDistributedSampler(val_dataset) # 验证集默认是打乱采样的，这对评估是好事（防止 overfit 某几个样本）
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        pin_memory=True,
    )
    # 创建一个固定样本 Dataset（只取 index=0）
    fixed_val_sample = val_dataset[0]  # Subset 支持索引
    fixed_val_video = fixed_val_sample["video"].clone()  # [C, T, H, W]

    # Optimizer #???
    modules_to_train = [module for module in model.module.get_decoder()]
    if not args.freeze_encoder:
        modules_to_train += [module for module in model.module.get_encoder()]
    else:
        for module in model.module.get_encoder():
            module.eval()
            module.requires_grad_(False)
        logger.warning("Encoder is freezed!")

    parameters_to_train = []
    for module in modules_to_train:
        parameters_to_train += list(filter(lambda p: p.requires_grad, module.parameters()))

    gen_optimizer = torch.optim.AdamW(parameters_to_train, lr=args.lr, weight_decay=1e-4)
    disc_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()), lr=args.lr, weight_decay=0.01
    )

    # 确认总参数量和可训练参数量
    total = sum(p.numel() for p in model.module.parameters())
    trainable = sum(p.numel() for p in model.module.parameters() if p.requires_grad)

    logging.info(f"Total parameters: {total:,}")
    logging.info(f"Trainable parameters: {trainable:,}")

    # AMP scaler; GradScaler 只在使用 float16 或 bfloat16 时才有意义
    scaler = torch.cuda.amp.GradScaler() # 在使用 float16/bfloat16 精度训练时，自动缩放梯度，避免数值精度溢出或下溢的问题
    precision = torch.bfloat16
    if args.mix_precision == "fp16":
        precision = torch.float16
    elif args.mix_precision == "fp32":
        precision = torch.float32
    logging.info(f"precision: {precision}")

    # Load from checkpoint
    start_epoch = 0
    current_step = 0
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            raise Exception(
                f"Make sure `{args.resume_from_checkpoint}` is a ckpt file."
            )
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        # logging.info("Before resume: %s", hash(tuple(p.data.cpu().numpy().flatten() for p in model.parameters())))
        logging.info(
            "Before resume: %s",
            hash(tuple(p.data.cpu().numpy().tobytes() for p in model.parameters()))
        )
        print(model.module.load_state_dict(checkpoint["state_dict"]["gen_model"], strict=False))
        # logging.info("After resume: %s", hash(tuple(p.data.cpu().numpy().flatten() for p in model.parameters())))
        logging.info(
            "After resume: %s",
            hash(tuple(p.data.cpu().numpy().tobytes() for p in model.parameters()))
        )
        disc.module.load_state_dict(checkpoint["state_dict"]["disc_model"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        gen_optimizer.load_state_dict(checkpoint["optimizer_state"]["gen_optimizer"])
        disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
        ddp_sampler.load_state_dict(checkpoint["sampler_state"])    
        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        logger.info(
            f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step}"
        )

    if args.ema: #?
        logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(model, args.ema_decay) # EMA 通常用于生成更平滑的推理模型
        ema.register()

    # Training log
    logger.info("Prepared!")
    dist.barrier()
    if global_rank == 0: #统计参数量
        gen_total = total_params(model.module)
        gen_encoder = total_params(model.module.encoder)
        gen_decoder = total_params(model.module.decoder)
        disc_total = total_params(disc.module)
        logger.info(f"=== Model Params ===")
        logger.info(f"Generator:\t\t{total_params(model.module)}M")
        logger.info(f"\t- Encoder:\t{total_params(model.module.encoder):d}M")
        logger.info(f"\t- Decoder:\t{total_params(model.module.decoder):d}M")
        logger.info(f"Discriminator:\t{total_params(disc.module):d}M")
        logger.info(f"===========")
        logger.info(f"Precision is set to: {args.mix_precision}!")
        logger.info("Start training!")
        wandb.log({
            "params/generator_total": gen_total,
            "params/generator_encoder": gen_encoder,
            "params/generator_decoder": gen_decoder,
            "params/discriminator_total": disc_total,
        }, step=0)

    # Training Bar
    bar_desc = ""
    bar = None
    if global_rank == 0:
        max_steps = (
            args.epochs * len(dataloader) if args.max_steps is None else args.max_steps
        )
        bar = tqdm.tqdm(total=max_steps, desc=bar_desc.format(current_epoch=0, loss=0)) # 把字符串 bar_desc 中的 {current_epoch} 和 {loss} 两个占位符，替换成 0 和 0，生成一个新的字符串
        # bar.update(current_step) #  这是 tqdm 的方法：update(n)
        safe_bar_update(bar)
        bar_desc = "Epoch: {current_epoch}, Loss: {loss}"
        logger.warning("Training Details: ")
        logger.warning(f" Max steps: {max_steps}")
        logger.warning(f" Dataset Samples: {len(dataloader)}")
        logger.warning(
            f" Total Batch Size: {args.batch_size} * {os.environ['WORLD_SIZE']}"
        )
    dist.barrier()

    # Training Loop
    num_epochs = args.epochs

    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, loss=f"-")
            safe_bar_update(bar)
            # bar.update()

    for epoch in range(num_epochs):
        set_train(modules_to_train)
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch
        # run_fixed_sample_visualization(model, dataset, rank, epoch, current_step, args)
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["video"].to(rank) # torch.Size([B, 3, 9, 256, 256])

            if (
                current_step % 2 == 1
                and current_step >= disc.module.discriminator_iter_start
            ):
                set_modules_requires_grad(modules_to_train, False)
                step_gen = False
                step_dis = True
            else:
                set_modules_requires_grad(modules_to_train, True)
                step_gen = True
                step_dis = False

            assert ( #??
                step_gen or step_dis
            ), "You should backward either Gen or Dis in a step."

            with torch.cuda.amp.autocast(dtype=precision):
                outputs = model(inputs)
                recon = outputs[0]
                posterior = outputs[1]
                if len(outputs) == 3: # which means there is wavelet output
                    wavelet_coeffs = outputs[2] if args.wavelet_loss else None
                else:
                    wavelet_coeffs = None

            # Generator Step
            if step_gen:
                with torch.cuda.amp.autocast(dtype=precision):
                    g_loss, g_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=0,
                        global_step=current_step,
                        last_layer=model.module.get_last_layer(),
                        wavelet_coeffs=wavelet_coeffs,
                        split="train",
                    )
                gen_optimizer.zero_grad()
                scaler.scale(g_loss).backward()
                # scaler.unscale_(gen_optimizer)
                # torch.nn.utils.clip_grad_norm_(parameters_to_train, 5e6)
                scaler.step(gen_optimizer)
                scaler.update()
                if args.ema:
                    ema.update()
                if global_rank == 0 and current_step % args.log_steps == 0: #wandb
                    wandb.log(
                        {"train/generator_loss": g_loss.item()}, step=current_step
                    )
                    wandb.log(
                        {"train/rec_loss": g_log['train/rec_loss']}, step=current_step
                    )
                    wandb.log(
                        {"train/latents_std": posterior.sample().std().item()}, step=current_step
                    )
                    if 'train/sb_loss' in g_log:
                        wandb.log(
                            {"train/sb_loss": g_log['train/sb_loss']}, step=current_step
                        )
                    if 'train/wl_loss' in g_log:
                        wandb.log(
                            {"train/wl_loss": g_log['train/wl_loss']}, step=current_step
                        )

            # Discriminator Step
            if step_dis:
                with torch.cuda.amp.autocast(dtype=precision):
                    d_loss, d_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=1,
                        global_step=current_step,
                        last_layer=None,
                        split="train",
                    )
                disc_optimizer.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(disc.module.discriminator.parameters(), 1.0)
                scaler.step(disc_optimizer)
                scaler.update()
                if global_rank == 0 and current_step % args.log_steps == 0: #wandb
                    wandb.log(
                        {"train/discriminator_loss": d_loss.item()}, step=current_step
                    )

            def valid_model(model, name="", fixed_val_video=None):
                set_eval(modules_to_train)
                psnr_list, lpips_list, video_log = valid(
                    global_rank, rank, model, val_dataloader, precision, args,
                    epoch=epoch,
                    step=current_step,
                )
                valid_psnr, valid_lpips, valid_video_log = gather_valid_result(
                    psnr_list, lpips_list, video_log, rank, dist.get_world_size()
                )
                if global_rank == 0:
                    name = "_" + name if name != "" else name
                    wandb.log( # 随机验证
                        {
                            f"val{name}/recon_random": wandb.Video(
                                np.array(valid_video_log), fps=1
                            )
                        },
                        step=current_step, # step=current_step 表示这次日志对应训练的哪个 step（用于曲线对齐）
                    )
                    wandb.log({f"val{name}/psnr": valid_psnr}, step=current_step) 
                    wandb.log({f"val{name}/lpips": valid_lpips}, step=current_step)
                    logger.info(f"{name} PSNR: {valid_psnr}")
                    logger.info(f"{name} LPIPS: {valid_lpips}")
                    logger.info(f"{name} Validation done.")

                    if fixed_val_video is not None:
                        with torch.no_grad(), torch.cuda.amp.autocast(dtype=precision):
                            input_video = fixed_val_video.unsqueeze(0).to(rank)
                            outputs = model(input_video)
                            recon_video = outputs[0][0].detach().cpu()
                            input_video = input_video[0].detach().cpu()

                            log_fixed_val_video(args.log_path, input_video, recon_video, current_step)

            if current_step % args.eval_steps == 0 or current_step == 0: # 验证
                if global_rank == 0:
                    logger.info("Starting validation...")
                valid_model(model, name="", fixed_val_video=fixed_val_video)
                # print("--------stop--------")
                if args.ema:
                    ema.apply_shadow()
                    valid_model(model, name="ema", fixed_val_video=fixed_val_video)
                    ema.restore()

            # Save Checkpoint
            if current_step % args.save_ckpt_step == 0 and global_rank == 0:
                file_path = save_checkpoint(
                    epoch,
                    current_step,
                    {
                        "gen_optimizer": gen_optimizer.state_dict(),
                        "disc_optimizer": disc_optimizer.state_dict(),
                    },
                    {
                        "gen_model": model.module.state_dict(),
                        "disc_model": disc.module.state_dict(),
                    },
                    scaler.state_dict(),
                    ddp_sampler.state_dict(),
                    ckpt_dir,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.shadow if args.ema else {},
                )
                logger.info(f"Checkpoint has been saved to `{file_path}`.")

            update_bar(bar)
            current_step += 1

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    # Exp setting
    parser.add_argument("--exp_name", type=str, default="test", help="number of epochs to train")
    parser.add_argument("--gen", type=str, default="opensoraplan", help="")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    # Training setting
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="number of epochs to train"
    )
    parser.add_argument("--save_ckpt_step", type=int, default=1000, help="")
    parser.add_argument("--ckpt_dir", type=str, default="./results/", help="")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--log_steps", type=int, default=5, help="log steps")
    parser.add_argument("--freeze_encoder", action="store_true", help="")
    parser.add_argument("--clip_grad_norm", type=float, default=1e5, help="")

    # Data
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--log_path", type=str, default="wandb/latest-run/files/media/videos/val_fixed/", help="")
    parser.add_argument("--num_frames", type=int, default=17, help="")
    parser.add_argument("--resolution", type=lambda s: tuple(map(int, s.strip("() ").split(','))) if ',' in s else int(s), default=256, help="Resolution as int or W,H tuple")    
    parser.add_argument("--sample_rate", type=int, default=2, help="")
    parser.add_argument("--dynamic_sample", action="store_true", help="")
    # Generator model
    parser.add_argument("--ignore_mismatched_sizes", action="store_true", help="")
    parser.add_argument("--find_unused_parameters", action="store_true", help="")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, help=""
    )
    parser.add_argument("--model_name", type=str, default=None, help="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="")
    parser.add_argument("--not_resume_training_process", action="store_true", help="")
    parser.add_argument("--model_config", type=str, default=None, help="")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="precision for training",
    )
    parser.add_argument("--wavelet_loss", action="store_true", help="")
    parser.add_argument("--wavelet_weight", type=float, default=0.1, help="")
    # Discriminator Model
    parser.add_argument("--load_disc_from_checkpoint", type=str, default=None, help="")
    parser.add_argument(
        "--disc_cls",
        type=str,
        default="opensora.models.causalvideovae.model.losses.LPIPSWithDiscriminator3D",
        help="",
    )
    parser.add_argument("--disc_start", type=int, default=5, help="")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="")
    parser.add_argument("--loss_type", type=str, default="l1", help="")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=1000, help="")
    parser.add_argument("--eval_video_path", type=str, default=None, help="")
    parser.add_argument("--eval_num_frames", type=int, default=17, help="")
    parser.add_argument("--eval_resolution", type=int, default=256, help="")
    parser.add_argument("--eval_sample_rate", type=int, default=1, help="")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="")
    parser.add_argument("--eval_subset_size", type=int, default=100, help="")
    parser.add_argument("--eval_num_video_log", type=int, default=4, help="")
    parser.add_argument("--eval_lpips", action="store_true", help="")

    # Dataset
    parser.add_argument("--dataset_num_worker", type=int, default=64, help="")

    # EMA
    parser.add_argument("--ema", action="store_true", help="")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="")

    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)


if __name__ == "__main__":
    # print("AutoencoderKLCogVideoX in globals:", 'AutoencoderKLCogVideoX' in globals()) # 当前全局命名空间
    main()
