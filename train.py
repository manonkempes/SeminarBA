import os
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LazyDataset(Dataset):
    def __init__(
        self,
        item_sales,
        categories,
        colors,
        fabrics,
        temporal_features,
        gtrends,
        img_paths,
        img_root,
        release_ord,
        product_id,
    ):
        self.item_sales = item_sales
        self.categories = categories
        self.colors = colors
        self.fabrics = fabrics
        self.temporal_features = temporal_features
        self.gtrends = gtrends
        self.img_paths = img_paths
        self.img_root = img_root
        self.release_ord = release_ord
        self.product_id = product_id

        self.transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.item_sales)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(os.path.join(self.img_root, img_path)).convert("RGB")
        img_tensor = self.transforms(img)

        return (
            self.item_sales[idx],
            self.categories[idx],
            self.colors[idx],
            self.fabrics[idx],
            self.temporal_features[idx],
            self.gtrends[idx],
            img_tensor,
            self.release_ord[idx],
            self.product_id[idx],
        )


class ZeroShotDataset:
    def __init__(self, data_df, img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len):
        self.data_df = data_df
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_root = img_root

    def _build_product_ids(self, data: pd.DataFrame) -> np.ndarray:
        if "global_product_id" in data.columns:
            return data["global_product_id"].astype(np.int64).values

        if "external_code" in data.columns:
            codes, _ = pd.factorize(data["external_code"])
            return codes.astype(np.int64)

        return np.arange(len(data), dtype=np.int64)

    def preprocess_data(self):
        data = self.data_df.copy()
        gtrends = []
        image_paths = []

        release_ord = pd.to_datetime(data["release_date"]).map(pd.Timestamp.toordinal).values
        product_id = self._build_product_ids(data)

        for _, row in tqdm(data.iterrows(), total=len(data), ascii=True):
            cat = row["category"]
            col = row["color"]
            fab = row["fabric"]
            start_date = row["release_date"]
            img_path = row["image_path"]

            gtrend_start = start_date - pd.DateOffset(weeks=52)

            cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[:self.trend_len]
            col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[:self.trend_len]
            fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[:self.trend_len]

            cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1, 1)).flatten()
            col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1, 1)).flatten()
            fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1, 1)).flatten()

            multitrends = np.vstack([cat_gtrend, col_gtrend, fab_gtrend])

            gtrends.append(multitrends)
            image_paths.append(img_path)

        gtrends = np.array(gtrends)

        drop_cols = ["external_code", "season", "release_date", "image_path"]
        if "global_product_id" in data.columns:
            drop_cols.append("global_product_id")

        existing_drop_cols = [col for col in drop_cols if col in data.columns]
        data.drop(existing_drop_cols, axis=1, inplace=True)

        item_sales = torch.FloatTensor(data.iloc[:, :12].values)
        temporal_features = torch.FloatTensor(data.iloc[:, 13:17].values)

        categories = torch.LongTensor([self.cat_dict[val] for val in data["category"].values])
        colors = torch.LongTensor([self.col_dict[val] for val in data["color"].values])
        fabrics = torch.LongTensor([self.fab_dict[val] for val in data["fabric"].values])

        gtrends = torch.FloatTensor(gtrends)
        release_ord = torch.LongTensor(release_ord)
        product_id = torch.LongTensor(product_id)

        return LazyDataset(
            item_sales=item_sales,
            categories=categories,
            colors=colors,
            fabrics=fabrics,
            temporal_features=temporal_features,
            gtrends=gtrends,
            img_paths=image_paths,
            img_root=self.img_root,
            release_ord=release_ord,
            product_id=product_id,
        )

    def get_loader(self, batch_size, train=True):
        print("Starting dataset creation process...")
        dataset = self.preprocess_data()

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=2,
        )

        print("Done.")
        return loader