import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm.auto import tqdm


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

TSV_DATA = Path(DATA_DIR) / "blank_dataset.tsv"


class TitleGenerator:
    def __init__(self):
        # トークナイザー（SentencePiece）
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True)

        # 学習済みモデル
        trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
        self.USE_GPU = torch.cuda.is_available()
        if self.USE_GPU:
            trained_model.cuda()
        trained_model.eval()

        self.trained_model = trained_model

        args_dict = dict(
            data_dir=DATA_DIR,
            model_name_or_path=PRETRAINED_MODEL_NAME,
            tokenizer_name_or_path=PRETRAINED_MODEL_NAME,
            n_gpu=1 if self.USE_GPU else 0,
            seed=42,
        )
        # 予測に用いるハイパーパラメータ
        args_dict.update(
            {
                # 'learning_rate':3e-4,
                # 'weight_decay':0.0,
                # 'adam_epsilon':1e-8,
                # 'warmup_steps':0,
                # 'gradient_accumulation_steps':1,
                # 'early_stop_callback':False,
                # 'fp_16':False,
                # 'opt_level':"O1",
                # 'max_grad_norm':1.0,
                "max_input_length": 512,  # 入力文の最大トークン数
                "max_target_length": 64,  # 出力文の最大トークン数
                "train_batch_size": 8,  # 訓練時のバッチサイズ
                "eval_batch_size": 8,  # テスト時のバッチサイズ
                "num_train_epochs": 8,  # 訓練するエポック数
            }
        )
        self.args = argparse.Namespace(**args_dict)

    @staticmethod
    def preprocess_body(text):
        return text.replace("\n", " ")

    def generate_title(self, body: str) -> list[str]:
        max_input_length = self.args.max_input_length
        max_target_length = self.args.max_target_length

        inputs = [self.preprocess_body(body)]
        batch = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        input_mask = batch["attention_mask"]
        if self.USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        outputs = self.trained_model.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            max_length=max_target_length,
            temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
            num_beams=10,  # ビームサーチの探索幅
            diversity_penalty=1.0,  # 生成結果の多様性を生み出すためのペナルティ
            num_beam_groups=10,  # ビームサーチのグループ数
            num_return_sequences=10,  # 生成する文の数
            repetition_penalty=1.5,  # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )

        # token -> str
        generated_titles = [
            self.tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for ids in outputs
        ]
        return generated_titles


if __name__ == "__main__":
    # titleを持たない条項本文情報
    with open(TSV_DATA) as fp:
        lines = [
            l.strip().split("\t")
            for l in fp.read().split("\n")
            if len(l.strip().split("\t")) == 3
        ]
        blank_data = [
            {"title": "", "body": body, "genre_id": i}
            for title, body, i in lines
            if len(body) > 0
        ]
    # predict
    tg = TitleGenerator()
    body_list = [d["body"] for d in blank_data]
    titles = [tg.generate_title(body) for body in tqdm(body_list)]

    export_path = Path(DATA_DIR) / "blank_predict.tsv"
    with open(export_path, "w") as fp:
        fp.write("generated\tbody\n")
        for candidates, input in zip(titles, body_list):
            output = ",".join(candidates)
            fp.write("\t".join([output, input]))
            fp.write("\n")