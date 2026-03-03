import os
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report, f1_score

from taaghche.config import DataConfig, TFConfig
from taaghche.data_prep import prepare_dataset
from taaghche.tf_pipeline.data import make_examples, batch_dataset


def build_model(model_name, config, learning_rate):
    model = TFBertForSequenceClassification.from_pretrained(model_name, config=config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model


def run():
    dcfg = DataConfig()
    tcfg = TFConfig()

    train_df, valid_df, test_df, labels, label2id, id2label = prepare_dataset(
        dcfg.csv_path, dcfg.min_words, dcfg.max_words, dcfg.label_threshold, dcfg.balance, dcfg.seed
    )

    tokenizer = BertTokenizer.from_pretrained(tcfg.model_name)
    config = BertConfig.from_pretrained(
        tcfg.model_name, label2id=label2id, id2label=id2label, num_labels=len(labels)
    )

    train_ds, train_examples = make_examples(tokenizer, train_df['comment'].tolist(), train_df['label'].tolist(), maxlen=tcfg.max_len)
    valid_ds, valid_examples = make_examples(tokenizer, valid_df['comment'].tolist(), valid_df['label'].tolist(), maxlen=tcfg.max_len)
    test_ds, test_examples = make_examples(tokenizer, test_df['comment'].tolist(), test_df['label'].tolist(), maxlen=tcfg.max_len)
    xtest, ytest = make_examples(tokenizer, test_df['comment'].tolist(), test_df['label'].tolist(), maxlen=tcfg.max_len, is_tf_dataset=False)[0]

    train_ds = batch_dataset(train_ds, tcfg.batch_size, shuffle=True).repeat()
    valid_ds = batch_dataset(valid_ds, tcfg.batch_size)

    train_steps = len(train_examples) // tcfg.batch_size
    valid_steps = len(valid_examples) // tcfg.batch_size

    model = build_model(tcfg.model_name, config, tcfg.learning_rate)
    model.fit(train_ds, validation_data=valid_ds, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=tcfg.epochs, verbose=1)

    eval_result = model.evaluate(batch_dataset(test_ds, tcfg.batch_size))
    print(f'eval: {eval_result}')

    preds = model.predict(xtest)[0].argmax(axis=-1).tolist()
    print(classification_report(ytest, preds, target_names=labels))
    print(f'F1: {f1_score(ytest, preds, average= weighted):.4f}')

    os.makedirs('artifacts', exist_ok=True)
    model.save_pretrained('artifacts/bert-fa-base-uncased-sentiment-taaghche-tf')


if __name__ == '__main__':
    run()
