import os
import argparse
import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime

from models.GTM import GTM
from models.FCN import FCN
from models.retrieval_gtm import RetrievalGTM
from utils.data_multitrends import ZeroShotDataset
from utils.retrieval_bank import RetrievalBank

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_retrieval_bank(model, loader, device):
    """
    Build retrieval bank from the subtrain set only.
    Assumes each batch is:
    (item_sales, category, color, fabric, temporal_features, gtrends, images, release_ord, product_id)
    """
    model.eval()

    bank_z = []
    bank_g = []
    bank_y = []
    bank_release_ord = []
    bank_product_id = []

    with torch.no_grad():
        for batch in loader:
            item_sales, category, color, fabric, temporal_features, gtrends, images, release_ord, product_id = batch

            item_sales = item_sales.to(device)
            category = category.to(device)
            color = color.to(device)
            fabric = fabric.to(device)
            temporal_features = temporal_features.to(device)
            gtrends = gtrends.to(device)
            images = images.to(device)
            release_ord = release_ord.to(device)
            product_id = product_id.to(device)

            z = model.encode_static(
                category=category,
                color=color,
                fabric=fabric,
                temporal_features=temporal_features,
                images=images,
            )

            _, g = model.encode_trends(gtrends)

            bank_z.append(z)
            bank_g.append(g)
            bank_y.append(item_sales)
            bank_release_ord.append(release_ord)
            bank_product_id.append(product_id)

    bank = RetrievalBank(
        z=torch.cat(bank_z, dim=0),
        g=torch.cat(bank_g, dim=0),
        y=torch.cat(bank_y, dim=0),
        release_ord=torch.cat(bank_release_ord, dim=0),
        product_id=torch.cat(bank_product_id, dim=0),
    )

    model.train()
    return bank


def run(args):
    print(args)

    # Seeds for reproducibility
    pl.seed_everything(args.seed)

    # Load sales data
    train_df = pd.read_csv(Path(args.data_folder + 'train.csv'), parse_dates=['release_date'])

    # Load category / color / fabric encodings
    cat_dict = torch.load(Path(args.data_folder + 'category_labels.pt'), weights_only=False)
    col_dict = torch.load(Path(args.data_folder + 'color_labels.pt'), weights_only=False)
    fab_dict = torch.load(Path(args.data_folder + 'fabric_labels.pt'), weights_only=False)

    # Load Google trends
    gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)

    # Sort on release date
    train_df = train_df.sort_values("release_date").reset_index(drop=True)

    # 85% subtrain / 15% validation split by time
    val_size = max(1, int(0.15 * len(train_df)))
    subtrain_df = train_df.iloc[:-val_size].copy()
    val_df = train_df.iloc[-val_size:].copy()

    train_dataset_builder = ZeroShotDataset(
        subtrain_df,
        Path(args.data_folder + '/images'),
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len
    )

    val_dataset_builder = ZeroShotDataset(
        val_df,
        Path(args.data_folder + '/images'),
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len
    )

    train_loader = train_dataset_builder.get_loader(
        batch_size=args.batch_size,
        train=True
    )

    val_loader = val_dataset_builder.get_loader(
        batch_size=1,
        train=False
    )

    # Separate non-shuffled loader for retrieval bank construction
    retrieval_bank_loader = train_dataset_builder.get_loader(
        batch_size=args.batch_size,
        train=False
    )

    # ---------------------------
    # Create model
    # ---------------------------
    if args.model_type == 'FCN':
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_trends=args.use_trends,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num
        )

    elif args.model_type == 'RetrievalGTM':
        model = RetrievalGTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            gpu_num=args.gpu_num,
            topk=args.retrieval_topk,
            retrieval_dim=args.retrieval_dim,
            retrieval_dropout=args.retrieval_dropout,
        )

    else:
        model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            gpu_num=args.gpu_num
        )

    # ---------------------------
    # Build retrieval bank if needed
    # ---------------------------
    if args.model_type == 'RetrievalGTM':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu_num}')
        else:
            device = torch.device('cpu')

        model = model.to(device)

        print('Building retrieval bank from subtrain set...')
        retrieval_bank = build_retrieval_bank(model, retrieval_bank_loader, device)
        model.set_retrieval_bank(retrieval_bank)
        print('Done building retrieval bank.')

    # ---------------------------
    # Model training
    # ---------------------------
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model_savename = args.model_type + '_' + args.wandb_run

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + '/' + args.model_type,
        filename=model_savename + '---{epoch}---' + dt_string,
        monitor='val_mae',
        mode='min',
        save_top_k=1
    )

    # wandb.init(entity=args.wandb_entity, project=args.wandb_proj, name=args.wandb_run)
    # wandb_logger = pl_loggers.WandbLogger()
    # wandb_logger.watch(model)

    tb_logger = pl_loggers.TensorBoardLogger(args.log_dir + '/', name=model_savename)

    trainer = pl.Trainer(
        gpus=[args.gpu_num],
        max_epochs=args.epochs,
        check_val_every_n_epoch=5,
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=0)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM',
                        help='Choose between GTM, FCN or RetrievalGTM')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    # Retrieval arguments
    parser.add_argument('--retrieval_topk', type=int, default=5)
    parser.add_argument('--retrieval_dim', type=int, default=64)
    parser.add_argument('--retrieval_dropout', type=float, default=0.1)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='username-here')
    parser.add_argument('--wandb_proj', type=str, default='GTM')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    run(args)