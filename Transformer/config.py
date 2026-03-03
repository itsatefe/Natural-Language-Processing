from dataclasses import dataclass
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent

@dataclass
class DataConfig:
    csv_path: Path = ROOT / 'data' / 'taghche.csv'
    min_words: int = 3
    max_words: int = 256
    balance: bool = True
    label_threshold: float = 3.0
    seed: int = 1

@dataclass
class TrainConfig:
    model_name: str = 'HooshvareLab/bert-fa-base-uncased'
    max_len: int = 128
    train_batch_size: int = 16
    valid_batch_size: int = 16
    test_batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    clip: float = 0.0
    output_path: Path = ROOT / 'artifacts' / 'pt_model.bin'
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class TFConfig:
    model_name: str = 'HooshvareLab/bert-fa-base-uncased'
    max_len: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
