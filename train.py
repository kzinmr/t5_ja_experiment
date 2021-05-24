import argparse
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)


# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# 事前学習済みモデル
PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"

# 転移学習済みモデルのディレクトリ
MODEL_DIR = "/app/workspace/model"
# データセットのディレクトリ
DATA_DIR = "/app/workspace/data"
assert os.path.exists(DATA_DIR)
assert os.path.exists(MODEL_DIR)

TSV_DATA = Path(DATA_DIR) / "dataset.tsv"

# GPU利用有無
USE_GPU = torch.cuda.is_available()


class TsvDataset(Dataset):
    def __init__(
        self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512
    ):
        self.file_path = os.path.join(data_dir, type_path)

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

    def _make_record(self, title, body, genre_id):
        input = f"{body}"
        target = f"{title}"
        return input, target

    def _build(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                assert len(line) == 3
                assert len(line[0]) > 0
                assert len(line[1]) > 0
                assert len(line[2]) > 0

                title = line[0]
                body = line[1]
                genre_id = line[2]

                input, target = self._make_record(title, body, genre_id)

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input],
                    max_length=self.input_max_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target],
                    max_length=self.target_max_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        self.tokenizer: T5Tokenizer
        self.train_dataset: TsvDataset
        self.val_dataset: TsvDataset
        self.test_dataset: TsvDataset

        super().__init__()
        self.hparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(
            hparams.model_name_or_path
        )

        self.tokenizer = T5Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, is_fast=True
        )

        self.all_data = self._read_tsv(TSV_DATA)

    @staticmethod
    def _read_tsv(tsv_path):
        with open(tsv_path) as fp:
            lines = [
                l.strip().split("\t")
                for l in fp.read().split("\n")
                if len(l.strip().split("\t")) == 3
            ]
            all_data = []
            seen = set()  # 厳密な重複の除去 # TODO: approx dedup
            for title, body, i in lines:
                if len(title) > 0 and len(body) > 0:
                    if body not in seen:
                        all_data.append({"title": title, "body": body, "genre_id": i})
                        seen.add(body)
        return all_data

    @staticmethod
    def _to_line(data):
        title = data["title"]
        body = data["body"]
        genre_id = data["genre_id"]

        assert len(title) > 0 and len(body) > 0
        return f"{title}\t{body}\t{genre_id}\n"

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            all_data = self.all_data
            data_size = len(all_data)
            train_ratio, dev_ratio, test_ratio = 0.9, 0.05, 0.05
            random.shuffle(all_data)

            train_data_path = Path(DATA_DIR) / "train.tsv"
            val_data_path = Path(DATA_DIR) / "dev.tsv"
            test_data_path = Path(DATA_DIR) / "test.tsv"
            with open(train_data_path, "w", encoding="utf-8") as f_train, open(
                val_data_path, "w", encoding="utf-8"
            ) as f_dev, open(test_data_path, "w", encoding="utf-8") as f_test:
                for i, data in tqdm(enumerate(all_data)):
                    line = self._to_line(data)
                    if i < train_ratio * data_size:
                        f_train.write(line)
                    elif i < (train_ratio + dev_ratio) * data_size:
                        f_dev.write(line)
                    else:
                        f_test.write(line)

            self.train_dataset = self.get_dataset(
                tokenizer=self.tokenizer, type_path="train.tsv", args=self.hparams
            )

            self.val_dataset = self.get_dataset(
                tokenizer=self.tokenizer, type_path="dev.tsv", args=self.hparams
            )

            self.test_dataset = self.get_dataset(
                tokenizer=self.tokenizer, type_path="test.tsv", args=self.hparams
            )

            self.t_total = (
                (
                    len(self.train_dataset)
                    // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu))
                )
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        # All labels set to -100 are ignored (masked),
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch["target_mask"],
            labels=labels,
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.t_total,
        )
        self.scheduler = scheduler

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def get_dataset(self, tokenizer, type_path, args):
        return TsvDataset(
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            type_path=type_path,
            input_max_len=args.max_input_length,
            target_max_len=args.max_target_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4
        )


def export_test_prediction(test_loader):
    """ 予測結果のexport """

    export_path = Path(DATA_DIR) / "test_predict.tsv"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True)
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    if USE_GPU:
        trained_model.cuda()
    trained_model.eval()

    inputs = []
    outputs = []
    targets = []
    for batch in tqdm(test_loader):
        input_ids = batch["source_ids"]
        input_mask = batch["source_mask"]
        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
        output = trained_model.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            max_length=args.max_target_length,
            temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
            repetition_penalty=1.5,  # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )
        output_text = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for ids in output
        ]
        target_text = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for ids in batch["target_ids"]
        ]
        input_text = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for ids in batch["source_ids"]
        ]

        inputs.extend(input_text)
        outputs.extend(output_text)
        targets.extend(target_text)

    with open(export_path, "w") as fp:
        fp.write("generated\tactual\tbody\n")
        for output, target, input in zip(outputs, targets, inputs):
            fp.write("\t".join([output, target, input]))
            fp.write("\n")


if __name__ == "__main__":
    args_dict = dict(
        data_dir=DATA_DIR,
        model_name_or_path=PRETRAINED_MODEL_NAME,
        tokenizer_name_or_path=PRETRAINED_MODEL_NAME,
        n_gpu=1 if USE_GPU else 0,
        seed=42,
    )
    # 学習に用いるハイパーパラメータ
    args_dict.update(
        {
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "adam_epsilon": 1e-8,
            "warmup_steps": 0,
            "gradient_accumulation_steps": 1,
            "early_stop_callback": False,
            "fp_16": False,
            "opt_level": "O1",
            "max_grad_norm": 1.0,
            "max_input_length": 512,  # 入力文の最大トークン数
            "max_target_length": 64,  # 出力文の最大トークン数
            "train_batch_size": 8,
            "eval_batch_size": 8,
            "num_train_epochs": 8,
        }
    )
    args = argparse.Namespace(**args_dict)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
    )

    # train（10min / epoch with GPU）
    module = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(module)

    # save model at the last epoch
    module.tokenizer.save_pretrained(MODEL_DIR)
    module.model.save_pretrained(MODEL_DIR)

    # predict on test
    export_test_prediction(module.test_dataloader())

    # eval on test
    trainer.test()
