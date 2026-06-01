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

from pathlib import Path
import os
import Levenshtein

from pathlib import Path
import os


print(f"Current working directory: {Path.cwd()}")

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
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=pixel_values.device
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


device = "cuda" if torch.cuda.is_available() else "cpu"
modeleval = Seq2SeqOCR().to(device)
modeleval = load_components(
    modeleval,
    "Daniel/Model_weights/final_model_weights",
    load_encoder=True,
    load_decoder=True,
    device=device
)

modeleval.eval()

tokenizer = CharTokenizer()

test_df = pd.read_csv("data/OCRcropsv3/test/tradOutputTest/annotatedTestV3.csv").dropna(subset=["annotated_transcription"])

test_dataset = OCRDataset("data/OCRcropsv3/test/tradOutputTest/annotatedTestV3.csv", IMG_DIR, tokenizer, augment=False, split="test/")


greedy_correct = beam_correct = 0
total = len(test_dataset)


results = []

for i in range(total):
    sample = test_dataset[i]
    pv     = sample["pixel_values"].unsqueeze(0).to(device)
    true   = test_df.iloc[i]["annotated_transcription"]

    greedy = modeleval.greedy_decode(pv)
    beam   = modeleval.beam_decode(pv, beam_width=10)

    results.append({
        "true": true,
        "greedy": greedy,
        "beam": beam,
        "image_name": test_df.iloc[i]["image"]
    })




class evaluation:
    ### functions for eval metrics, input = (gt, model output list) (ensure order)
    def __init__(self, groundTruth, results):
        self.gt = groundTruth
        self.pred = results

    def accuracy(self):
        correct = (self.gt == self.pred)
        return correct.mean() 

    def character_accuracy(self):
        total_chars = 0
        total_errors = 0

        for gt, pred in zip(self.gt, self.pred):
            gt = str(gt)
            pred = str(pred)

            dist = Levenshtein.distance(gt, pred)

            total_errors += dist
            total_chars += len(gt)

        return 1 - (total_errors / total_chars)
    



df_results = pd.DataFrame(results)

greedy_evaluation = evaluation(df_results["true"], df_results["greedy"])
beam_evaluation = evaluation(df_results["true"], df_results["beam"])

print(f"Greedy Word Accuracy: {greedy_evaluation.accuracy()}")
print(f"Greedy Character Accuracy: {greedy_evaluation.character_accuracy()}")


print(f"Beam Word Accuracy: {beam_evaluation.accuracy()}")
print(f"Beam Character Accuracy: {beam_evaluation.character_accuracy()}")
