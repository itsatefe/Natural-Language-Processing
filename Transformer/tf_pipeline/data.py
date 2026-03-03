import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from transformers import glue_convert_examples_to_features

class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def make_examples(tokenizer, x, y=None, maxlen=128, output_mode='classification', is_tf_dataset=True):
    y = y if isinstance(y, (list, np.ndarray)) else [None] * len(x)
    examples = []
    for i, (_x, _y) in tqdm(enumerate(zip(x, y)), total=len(x), desc='examples'):
        guid = str(i)
        label = int(_y) if _y is not None else None
        if isinstance(_x, str):
            text_a, text_b = _x, None
        else:
            text_a, text_b = _x
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    features = glue_convert_examples_to_features(
        examples, tokenizer, maxlen, output_mode=output_mode, label_list=list(np.unique(y))
    )

    if is_tf_dataset:
        all_input_ids = tf.constant([f.input_ids for f in features])
        all_attention_masks = tf.constant([f.attention_mask for f in features])
        all_token_type_ids = tf.constant([f.token_type_ids for f in features])
        all_labels = tf.constant([f.label for f in features])
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_masks,
                'token_type_ids': all_token_type_ids,
                'labels': all_labels,
            }
        )
        return dataset, features

    xdata = [
        np.array([f.input_ids for f in features]),
        np.array([f.attention_mask for f in features]),
        np.array([f.token_type_ids for f in features]),
    ]
    ydata = np.array([f.label for f in features])
    return [xdata, ydata], features

def batch_dataset(ds, batch_size, shuffle=False):
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(batch_size)
