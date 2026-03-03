import collections
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AdamW, BertConfig, BertTokenizer, get_linear_schedule_with_warmup

from taaghche.config import DataConfig, TrainConfig
from taaghche.data_prep import prepare_dataset
from taaghche.torch_pipeline.data import create_data_loader
from taaghche.torch_pipeline.model import SentimentModel


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device, clip):
    model.train()
    losses, preds, targets = [], [], []
    for batch in tqdm(data_loader, desc='train'):
        batch = to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
        loss = loss_fn(outputs, batch['targets'])
        loss.backward()
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        preds.append(outputs.argmax(1).detach().cpu())
        targets.append(batch['targets'].detach().cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return np.mean(losses), preds, targets


def eval_epoch(model, data_loader, loss_fn, device):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='val'):
            batch = to_device(batch, device)
            outputs = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            loss = loss_fn(outputs, batch['targets'])
            losses.append(loss.item())
            preds.append(outputs.argmax(1).cpu())
            targets.append(batch['targets'].cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return np.mean(losses), preds, targets


def f1_macro(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='weighted')


def run():
    dcfg = DataConfig()
    tcfg = TrainConfig()

    train_df, valid_df, test_df, labels, label2id, id2label = prepare_dataset(
        dcfg.csv_path, dcfg.min_words, dcfg.max_words, dcfg.label_threshold, dcfg.balance, dcfg.seed
    )

    tokenizer = BertTokenizer.from_pretrained(tcfg.model_name)
    config = BertConfig.from_pretrained(
        tcfg.model_name, label2id=label2id, id2label=id2label, num_labels=len(labels)
    )

    train_loader = create_data_loader(
        train_df['comment'].to_numpy(),
        train_df['label'].to_numpy(),
        tokenizer,
        tcfg.max_len,
        tcfg.train_batch_size,
        labels,
    )
    valid_loader = create_data_loader(
        valid_df['comment'].to_numpy(),
        valid_df['label'].to_numpy(),
        tokenizer,
        tcfg.max_len,
        tcfg.valid_batch_size,
        labels,
    )
    test_loader = create_data_loader(
        test_df['comment'].to_numpy(),
        test_df['label'].to_numpy(),
        tokenizer,
        tcfg.max_len,
        tcfg.test_batch_size,
        labels,
    )

    model = SentimentModel(tcfg.model_name, config).to(tcfg.device)
    optimizer = AdamW(model.parameters(), lr=tcfg.learning_rate, correct_bias=False)
    total_steps = len(train_loader) * tcfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf')
    history = collections.defaultdict(list)

    for epoch in range(1, tcfg.epochs + 1):
        train_loss, train_pred, train_true = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, tcfg.device, tcfg.clip
        )
        val_loss, val_pred, val_true = eval_epoch(model, valid_loader, loss_fn, tcfg.device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(f1_macro(train_true, train_pred))
        history['val_f1'].append(f1_macro(val_true, val_pred))

        if val_loss < best_val:
            tcfg.output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), tcfg.output_path)
            best_val = val_loss

    test_loss, test_pred, test_true = eval_epoch(model, test_loader, loss_fn, tcfg.device)
    print(f'test_loss={test_loss:.4f} test_f1={f1_macro(test_true, test_pred):.4f}')
    return history


if __name__ == '__main__':
    run()
