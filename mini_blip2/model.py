from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import CLIPVisionModel, OPTForCausalLM


@dataclass
class MiniBlip2Config:
    vision_model_name: str = "openai/clip-vit-base-patch32"
    language_model_name: str = "facebook/opt-125m"
    num_query_tokens: int = 16
    qformer_hidden_size: int = 256
    qformer_layers: int = 2
    qformer_heads: int = 8
    qformer_dropout: float = 0.1


class MiniQFormerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        self_out, _ = self.self_attn(query_tokens, query_tokens, query_tokens)
        query_tokens = self.norm1(query_tokens + self.dropout(self_out))

        cross_out, _ = self.cross_attn(query_tokens, image_tokens, image_tokens)
        query_tokens = self.norm2(query_tokens + self.dropout(cross_out))

        ffn_out = self.ffn(query_tokens)
        query_tokens = self.norm3(query_tokens + self.dropout(ffn_out))
        return query_tokens


class MiniQFormer(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        hidden_size: int,
        num_query_tokens: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size) * 0.02)
        self.vision_projection = nn.Linear(vision_hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            [MiniQFormerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = image_tokens.shape[0]
        image_tokens = self.vision_projection(image_tokens)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            query_tokens = layer(query_tokens, image_tokens)
        return self.final_norm(query_tokens)


class MiniBlip2ForCaptioning(nn.Module):
    def __init__(self, config: MiniBlip2Config | None = None) -> None:
        super().__init__()
        self.config = config or MiniBlip2Config()
        self.vision_encoder = CLIPVisionModel.from_pretrained(self.config.vision_model_name)
        self.language_decoder = OPTForCausalLM.from_pretrained(self.config.language_model_name)

        vision_hidden = self.vision_encoder.config.hidden_size
        lm_hidden = self.language_decoder.config.word_embed_proj_dim

        self.qformer = MiniQFormer(
            vision_hidden_size=vision_hidden,
            hidden_size=self.config.qformer_hidden_size,
            num_query_tokens=self.config.num_query_tokens,
            num_layers=self.config.qformer_layers,
            num_heads=self.config.qformer_heads,
            dropout=self.config.qformer_dropout,
        )
        self.language_projection = nn.Linear(self.config.qformer_hidden_size, lm_hidden)
        self.freeze_backbones()

    def freeze_backbones(self) -> None:
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.language_decoder.parameters():
            parameter.requires_grad = False
        self.vision_encoder.eval()
        self.language_decoder.eval()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def encode_image_prefix(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_tokens = vision_outputs.last_hidden_state
        qformer_outputs = self.qformer(image_tokens)
        prefix = self.language_projection(qformer_outputs)
        lm_dtype = self.language_decoder.model.decoder.embed_tokens.weight.dtype
        return prefix.to(dtype=lm_dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        prefix_embeds = self.encode_image_prefix(pixel_values)
        token_embeds = self.language_decoder.model.decoder.embed_tokens(input_ids)
        token_embeds = token_embeds.to(dtype=prefix_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_attention = torch.ones(
            prefix_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                prefix_embeds.shape[:2],
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)

        return self.language_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer,
        prompt: str = "",
        max_new_tokens: int = 30,
        num_beams: int = 3,
    ) -> torch.Tensor:
        self.eval()
        prefix_embeds = self.encode_image_prefix(pixel_values)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(pixel_values.device)
        prompt_embeds = self.language_decoder.model.decoder.embed_tokens(prompt_ids)
        inputs_embeds = torch.cat(
            [prefix_embeds, prompt_embeds.expand(prefix_embeds.shape[0], -1, -1)], dim=1
        )
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=pixel_values.device)
        return self.language_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
