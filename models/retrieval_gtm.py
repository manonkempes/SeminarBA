from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from models.GTM import GTM
from models.retrieval import RetrievalModule
from utils.retrieval_bank import RetrievalBank, build_retrieval_mask


class RetrievalGTM(GTM):
    def __init__(
        self,
        topk: int = 5,
        retrieval_dim: int = 64,
        retrieval_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.retrieval_module = RetrievalModule(
            prod_dim=self.hidden_dim,
            trend_dim=self.hidden_dim,
            horizon=self.output_len,
            retrieval_dim=retrieval_dim,
            topk=topk,
            dropout=retrieval_dropout,
        )

        self.retrieval_bank: Optional[RetrievalBank] = None

    def encode_static(
        self,
        category,
        color,
        fabric,
        temporal_features,
        images,
    ) -> torch.Tensor:
        img_encoding = self.image_encoder(images)
        time_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        z = self.static_feature_encoder(img_encoding, text_encoding, time_encoding)
        return z

    def encode_trends(self, gtrends) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            E: [T, B, D] full encoded trend sequence for decoder
            g: [B, D] pooled trend representation for retrieval
        """
        E = self.gtrend_encoder(gtrends)
        g = E.mean(dim=0)
        return E, g

    def decode_from_embedding(self, z_tilde: torch.Tensor, E: torch.Tensor):
        tgt = z_tilde.unsqueeze(0)  # [1, B, D]
        decoder_out, attn_weights = self.decoder(tgt, E)
        forecast = self.decoder_fc(decoder_out).squeeze(0)  # [B, H]
        return forecast, attn_weights

    def set_retrieval_bank(self, bank: RetrievalBank) -> None:
        self.retrieval_bank = bank

    def forward(
        self,
        category,
        color,
        fabric,
        temporal_features,
        gtrends,
        images,
        release_ord,
        product_id,
    ):
        if self.retrieval_bank is None:
            raise RuntimeError("Retrieval bank has not been set.")

        z = self.encode_static(category, color, fabric, temporal_features, images)
        E, g = self.encode_trends(gtrends)

        valid_mask = build_retrieval_mask(
            target_release_ord=release_ord,
            target_product_id=product_id,
            bank_release_ord=self.retrieval_bank.release_ord,
            bank_product_id=self.retrieval_bank.product_id,
            horizon_weeks=self.output_len,
        )

        z_tilde, retrieval_aux = self.retrieval_module(
            z_i=z,
            g_i=g,
            bank_z=self.retrieval_bank.z,
            bank_g=self.retrieval_bank.g,
            bank_y=self.retrieval_bank.y,
            valid_mask=valid_mask,
        )

        y_hat, attn_weights = self.decode_from_embedding(z_tilde, E)

        aux = {
            "retrieval": retrieval_aux,
            "attn_weights": attn_weights,
            "z": z,
            "z_tilde": z_tilde,
            "g": g,
        }
        return y_hat, aux

    def training_step(self, train_batch, batch_idx):
        (
            item_sales,
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            release_ord,
            product_id,
        ) = train_batch

        forecasted_sales, _ = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            release_ord,
            product_id,
        )

        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        (
            item_sales,
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            release_ord,
            product_id,
        ) = val_batch

        forecasted_sales, _ = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            release_ord,
            product_id,
        )

        return item_sales.squeeze(), forecasted_sales.squeeze()