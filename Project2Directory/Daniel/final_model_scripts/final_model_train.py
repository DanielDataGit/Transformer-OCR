import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


from pathlib import Path
import os
print(f"Current working directory: {Path.cwd()}")


# Save
def save_components(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Save encoder weights only
    encoder_state = {
        k: v for k, v in model.state_dict().items()
        if k.startswith("encoder")
    }
    torch.save(encoder_state, os.path.join(save_dir, "encoder.pt"))

    # Save decoder weights only 
    decoder_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("encoder")
    }
    torch.save(decoder_state, os.path.join(save_dir, "decoder.pt"))

    # Save full model too as a fallback
    torch.save(model.state_dict(), os.path.join(save_dir, "full_model.pt"))

    print(f"Saved encoder ({len(encoder_state)} tensors), "
          f"decoder ({len(decoder_state)} tensors), "
          f"full model to {save_dir}")


# Load
def load_components(model, save_dir, load_encoder=True, load_decoder=True, device="cpu"):
    if load_encoder:
        encoder_state = torch.load(
            os.path.join(save_dir, "encoder.pt"), map_location=device
        )
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        bad_missing = [k for k in missing if k.startswith("encoder")]
        print(f"Encoder loaded — {len(encoder_state)} tensors, "
              f"problematic missing: {bad_missing}")

    if load_decoder:
        decoder_state = torch.load(
            os.path.join(save_dir, "decoder.pt"), map_location=device
        )
        missing, unexpected = model.load_state_dict(decoder_state, strict=False)
        bad_missing = [k for k in missing if not k.startswith("encoder")]
        print(f"Decoder loaded — {len(decoder_state)} tensors, "
              f"problematic missing: {bad_missing}")

    return model


# configs for tokenizer and dataset paths 
chars      = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/"
PAD_ID     = 0
SOS_ID     = 1
EOS_ID     = 2
char2id    = {c: i + 3 for i, c in enumerate(chars)}
id2char    = {i + 3: c for i, c in enumerate(chars)}
VOCAB_SIZE = len(chars) + 3
MAX_LEN    = 9

TRAIN_CSV = "data/OCRcropsv3/train/tradOutputTrain/annotatedTrainV3.csv"
VAL_CSV   = "data/OCRcropsv3/val/tradOutputVal/annotatedValV3.csv"
IMG_DIR   = "data/OCRcropsv3"

# handle character set tokenization 
class CharTokenizer:
    def encode(self, text):
        return [char2id[c] for c in text if c in char2id]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in id2char:
                out.append(id2char[i])
        return "".join(out)

# augmentation and transforms for image preprocessing and training 
train_augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.05, hue=0.03),
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.92, 1.08), shear=3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomAutocontrast(p=0.2),
])


base_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.rotate(90, expand=True)),
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# dataloader class for OCR dataset 
class OCRDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, max_length=MAX_LEN, augment=False, split = "train/"):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["annotated_transcription"].notna()].reset_index(drop=True)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.split = split
        self.df["relative_path"] = self.df["image"].str.replace(r".*\?d=[^/]+/", "", regex=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, self.split, row["relative_path"])).convert("RGB")
        if self.augment:
            image = train_augment(image)

        pixel_values = base_transform(image)

        ids = self.tokenizer.encode(str(row["annotated_transcription"]))[:self.max_length]
        decoder_input = torch.tensor([SOS_ID] + ids, dtype=torch.long)
        labels = torch.tensor(ids + [EOS_ID], dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "decoder_input": decoder_input,
            "labels": labels,
        }

 

# collator function to handle padding of variable-length decoder inputs and labels in the dataloader 
def ocr_collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    max_tgt = max(b["decoder_input"].size(0) for b in batch)
    dec_inputs, label_list = [], []

    for b in batch:
        pad = max_tgt - b["decoder_input"].size(0)
        dec_inputs.append(torch.cat([
            b["decoder_input"],
            torch.full((pad,), PAD_ID, dtype=torch.long)
        ]))
        label_list.append(torch.cat([
            b["labels"],
            torch.full((pad,), -100, dtype=torch.long)
        ]))

    return {
        "pixel_values": pixel_values,
        "decoder_input": torch.stack(dec_inputs),
        "labels": torch.stack(label_list),
    }


# class to handle sinusoidal positional encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
    

# CRNN module 
class OCRCNNEncoder(nn.Module):
    def __init__(self, d_model=384):
        super().__init__()

        self.cnn = nn.Sequential(
            # input is (B, 3, 64, 256) 
            nn.Conv2d(3, 64, 3, 1, 1),   
            nn.ReLU(inplace=True),       
            nn.MaxPool2d(2, 2), #(B, 64, 32, 128)       

            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.ReLU(inplace=True),       
            nn.MaxPool2d(2, 2), #(B, 128, 16, 64)          

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),  #(B, 256, 16, 64)       

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),       
            nn.MaxPool2d((2,1), (2,1)), # (B, 256, 8, 64)  

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),       
            nn.MaxPool2d((2,1), (2,1)), # (B, 256, 4, 64) 
        )

        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        self.proj = nn.Linear(512, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.cnn(x) # (B, 256, 4, 64) 

        # Add dropout
        x = F.dropout2d(x, p=0.1, training=self.training)

        x = x.mean(dim=2) # collapse height to 1-d (B, 256, 64)
        x = x.permute(0, 2, 1) # (B, 64, 256) 

        x, _ = self.bilstm(x)  # (B, 64, 512) 

        # Add dropout after BiLSTM
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.proj(x) #(B, 64, d_model)
        x = self.norm(x)

        return x

# Combined model class that integrates the CNN encoder and Transformer decoder for OCR 
class Seq2SeqOCR(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=384,
        nhead=8,
        num_dec_layers=2,
        dim_feedforward=768,
        dropout=0.3,
    ):
        super().__init__()

        # OCR CRNN encoder 
        self.encoder = OCRCNNEncoder(d_model=d_model)
        self.enc_proj = nn.Identity()  

        # Seq to Seq Decoder
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.15)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,}  ({100*trainable/total:.1f}%)")

    def encode_image(self, pixel_values):
        return self.encoder(pixel_values)

    def forward(self, pixel_values, decoder_input, labels=None):
        memory = self.encode_image(pixel_values) #returns crnn output (B, 64, d_model = 384) 


        tgt_len = decoder_input.size(1) #length of decoder input sequence
        #causal mask to prevent attending to future tokens 
        tgt_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=pixel_values.device, dtype=torch.bool),
            diagonal=1
        )
        tgt_key_padding_mask = (decoder_input == PAD_ID) #boolean mask to prevent attending to padding tokens 

        tgt = self.pos_enc(self.tok_embed(decoder_input)) #token embedding + positional encoding for decoder input 

        #transformer decoder that attends to encoder output (memory) and processes the target sequence (tgt) with appropriate masks 
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.out_proj(out) #generate scores per token 

        loss = None
        if labels is not None:
            #compute loss by comparing logits to labels
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

        return {"loss": loss, "logits": logits}


# Inference methods for greedy and beam search decoding 
    @torch.no_grad()
    def greedy_decode(self, pixel_values, max_len=MAX_LEN):
        self.eval()
        memory = self.encode_image(pixel_values) #returns crnn output (B, 64, d_model = 384) 
        dec_input = torch.tensor([[SOS_ID]], device=pixel_values.device) #init with sos token 

        for _ in range(max_len): #autoregressive decoding loop
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                dec_input.size(1), device=pixel_values.device
            )
            tgt = self.pos_enc(self.tok_embed(dec_input))
            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            next_tok = self.out_proj(out[:, -1, :]).argmax(-1, keepdim=True) #choose token with highest score as next token (greedy) 

            if next_tok.item() == EOS_ID:
                break

            dec_input = torch.cat([dec_input, next_tok], dim=1)

        ids = dec_input[0, 1:].tolist()
        return "".join(id2char[i] for i in ids if i in id2char) #returns characters 

    @torch.no_grad()
    def beam_decode(self, pixel_values, beam_width=5, max_len=MAX_LEN):
        self.eval()
        memory = self.encode_image(pixel_values)
        beams = [(0.0, [SOS_ID])] #init beams
        completed = []

        for _ in range(max_len):
            candidates = []

            for score, ids in beams:
                if ids[-1] == EOS_ID:
                    completed.append((score, ids))
                    continue

                dec_input = torch.tensor([ids], device=pixel_values.device)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    len(ids), device=pixel_values.device
                )
                tgt = self.pos_enc(self.tok_embed(dec_input))
                out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
                log_prob = self.out_proj(out[:, -1, :]).log_softmax(-1)[0]
                topk = log_prob.topk(beam_width)

                for lp, tok in zip(topk.values, topk.indices):
                    candidates.append((score + lp.item(), ids + [tok.item()]))

            if not candidates:
                break

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

            if all(b[1][-1] == EOS_ID for b in beams):
                completed.extend(beams)
                break

        if not completed:
            completed = beams

        best = max(completed, key=lambda x: x[0])[1]
        ids = [i for i in best[1:] if i != EOS_ID]
        return "".join(id2char[i] for i in ids if i in id2char)


# Custom Trainer class to handle training loop and compute loss for the OCR model, with added logging of learning rate every 50 steps 
class OCRTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            pixel_values=inputs["pixel_values"],
            decoder_input=inputs["decoder_input"],
            labels=inputs["labels"],
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(), "lr": 1e-4},
                {"params": self.model.tok_embed.parameters(), "lr": 1e-4},
                {"params": self.model.decoder.parameters(), "lr": 1e-4},
                {"params": self.model.out_proj.parameters(), "lr": 1e-4},
            ],
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer
    

# training setup 
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CharTokenizer()

train_dataset = OCRDataset(TRAIN_CSV, IMG_DIR, tokenizer, augment=True, split="train/")
val_dataset = OCRDataset(VAL_CSV, IMG_DIR, tokenizer, augment=False, split="val/")

# call to load in model weights
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Seq2SeqOCR().to(device)
model = load_components(
    model,
    "Daniel/Model_weights/stage1_sroie_weights",
    load_encoder=True,
    load_decoder=True,
    device=device
)


# HF trainer setup with early stopping 
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=150,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    weight_decay=0.05,
    warmup_steps=150,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
)

trainer = OCRTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=ocr_collate_fn,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=15,
            early_stopping_threshold=0.001,
        )
    ],
)

trainer.train()

save_components(model, "./final_modelv2")
                

logs = trainer.state.log_history

train_loss = [(l["epoch"], l["loss"]) for l in logs if "loss" in l and "eval_loss" not in l]
val_loss   = [(l["epoch"], l["eval_loss"]) for l in logs if "eval_loss" in l]

train_epochs, train_losses = zip(*train_loss)
val_epochs,   val_losses   = zip(*val_loss)

plt.figure(figsize=(10, 5))
plt.plot(train_epochs, train_losses, label="Training Loss")
plt.plot(val_epochs,   val_losses,   label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
