"""
models/text_encoder.py

将文本描述编码为条件序列，供去噪器 Cross-Attention 使用。

输入：List[str]，批量文本
输出：
  text_tokens: (B, L, text_out)  token 级别序列（供 cross-attn）
  text_cls:    (B, text_out)     [CLS] 全局嵌入（可选用于全局条件）
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    BERT 文本编码器 + 线性投影。

    - 默认冻结 BERT 权重（freeze_bert=True），只训练投影层
    - 若无 GPU 或测试环境，可用随机 MLP 替代（use_mock=True）

    Args:
        model_name (str): HuggingFace 模型名
        out_dim (int):    投影后的输出维度
        freeze (bool):    是否冻结 BERT
        max_length (int): 最长 token 数（截断/填充）
    """

    def __init__(self, model_name: str = "bert-base-uncased",
                 out_dim: int = 256, freeze: bool = True,
                 max_length: int = 64, use_mock: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.max_length = max_length
        self.use_mock = use_mock

        if use_mock:
            # 用于快速测试，不依赖 transformers
            self.mock_emb = nn.Embedding(30522, 768)
            bert_dim = 768
        else:
            from transformers import BertModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert = BertModel.from_pretrained(model_name)
            if freeze:
                for p in self.bert.parameters():
                    p.requires_grad = False
            bert_dim = self.bert.config.hidden_size  # 768

        # 投影：768 → out_dim，保留序列维度
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def encode_text(self, texts, device):
        """
        将文本列表编码为 BERT 隐状态序列。

        Returns:
            last_hidden: (B, L, 768)
            attention_mask: (B, L)  有效位置为 1
        """
        if self.use_mock:
            B = len(texts)
            L = self.max_length
            ids = torch.randint(0, 30522, (B, L), device=device)
            mask = torch.ones(B, L, device=device)
            hidden = self.mock_emb(ids)  # (B, L, 768)
            return hidden, mask

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.set_grad_enabled(not self.training):
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        # last_hidden_state: (B, L, 768)
        return outputs.last_hidden_state, attention_mask.float()

    def forward(self, texts, device=None):
        """
        Args:
            texts:  List[str]，长度 B
            device: torch.device

        Returns:
            tokens:  (B, L, out_dim)   token 序列（含填充位置）
            cls_emb: (B, out_dim)      [CLS] 全局嵌入
            mask:    (B, L)            有效位掩码
        """
        if device is None:
            device = next(self.parameters()).device

        hidden, mask = self.encode_text(texts, device)  # (B, L, 768)

        tokens = self.proj(hidden)                       # (B, L, out_dim)
        cls_emb = tokens[:, 0, :]                        # (B, out_dim)

        return tokens, cls_emb, mask
