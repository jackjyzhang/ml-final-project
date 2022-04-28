import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator, pipeline
from tqdm import tqdm
import wandb

import pickle
import argparse
import os
import logging
import math
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
random.seed(42)

class HFWrapperDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        super(HFWrapperDataset).__init__()
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def cmd_collate_fn(data, tokenizer):
    texts = [e['tweet'] for e in data]
    labels = [e['class'] for e in data]
    source = tokenizer(texts, padding="longest", truncation=True, return_tensors="pt", return_attention_mask=True)
    target = torch.LongTensor(labels)
    return {
        "input_ids": source["input_ids"],
        "attention_mask": source["attention_mask"],
        "labels": target,
    }


def torch_dataloader_from_hf(hf_dataset, batchsize, tokenizer):
    collate_fn = lambda data: cmd_collate_fn(data, tokenizer)
    return torch.utils.data.DataLoader(
        HFWrapperDataset(hf_dataset),
        batch_size=batchsize,
        collate_fn=collate_fn,
    )

def construct_dataloaders(hf_dataset_path, batchsize, tokenizer):
    dataset = load_from_disk(hf_dataset_path)
    train_loader = torch_dataloader_from_hf(dataset['train'], batchsize, tokenizer)
    val_loader = torch_dataloader_from_hf(dataset['val'], batchsize, tokenizer)
    test_loader = torch_dataloader_from_hf(dataset['test'], batchsize, tokenizer)
    return train_loader, val_loader, test_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default='roberta-large', help='huggingface model string')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1, help="Eval model every x epochs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test-eval", action="store_true", help="Evaluate model on test set.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--load-dir", type=str, default=None, help="Directory containing checkpoint pytorch_model.bin")
    parser.add_argument("--start-epoch", type=int, default=0, help="If load epoch_t checkpoint, then use t (next epoch)")
    parser.add_argument("--schedule", action="store_true", help="Use OneCycleLR")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="Skip training loop (also turns off wandb)")
    parser.add_argument("--num-labels", type=int, required=True)
    parser.add_argument("--print-per-num-batch", type=int, default=100)
    parser.add_argument("--loss-balance", type=float, nargs="*", default=None) # should be [1/n_class0, 1/n_class1, 1/n_class2]
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")

    # model initialization
    if args.load_dir is not None:
        model = AutoModelForSequenceClassification.from_pretrained(args.load_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.load_dir)
        stats_path = os.path.join(args.load_dir, "train_stats.pkl")
        with open(stats_path, "rb") as f:
            train_stats = pickle.load(f)
    else:
        # Not loading checkpoint. Initialize model from pretrained
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        train_stats = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
    model = model.to(device)

    # Initialize generation pipeline
    # https://huggingface.co/transformers/main_classes/pipelines.html#transformers.Text2TextGenerationPipeline
    # pipe = pipeline(
    #     "text-classification",
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=torch.cuda.current_device() if device == torch.device("cuda") else -1,
    # )

    # Load data
    train_loader, val_loader, test_loader = construct_dataloaders(args.data_dir, args.batchsize, tokenizer)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.schedule:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, 
            max_lr=args.learning_rate, 
            epochs=args.epochs-args.start_epoch, 
            steps_per_epoch=len(train_loader)
        )

    dataset_name = os.path.basename(args.data_dir)
    exp_dir_name = os.path.basename(args.output_dir)
    wandb.init(
        mode= 'disabled' if args.disable_wandb or args.no_train else 'offline',
        project='bertweet',
        name=exp_dir_name,
        config={
            'model_str': args.model,
            'dataset': dataset_name,
            'optimizer': optim.__class__.__name__,
            'batchsize': args.batchsize,
            'lr': args.learning_rate,
            'wt_decay': args.weight_decay,
            'epochs': args.epochs,
            'start_epoch': args.start_epoch,
            'onecyclelr': args.schedule
        }
    )

    if not args.no_train:
        if args.loss_balance is not None:
            loss_weights = torch.FloatTensor(args.loss_balance).to(device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=loss_weights)
        for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epoch", ncols=0, total=args.epochs-args.start_epoch):
            model.train()
            epoch_loss = 0
            n_correct = 0
            n_total = 0
            for b, batch in enumerate(tqdm(train_loader, desc="Batch", ncols=0, total=len(train_loader))):
                optim.zero_grad()
                input_ids, attention_mask, labels = [b.to(device) for _,b in batch.items()]
                outputs = model(input_ids, attention_mask, labels=labels)
                if args.loss_balance is not None:
                    loss = loss_fct(outputs.logits.view(-1, args.num_labels), labels.view(-1))
                else:    
                    loss = outputs.loss
                epoch_loss += loss.item()
                loss.backward()
                optim.step()
                if args.schedule: scheduler.step()

                n_correct += torch.eq(outputs.logits.detach().max(dim=1).indices, labels).sum().cpu().item()
                n_total += labels.size(0)

                if b % args.print_per_num_batch == 0:
                    logger.info(f'loss: {loss.item()}, loss_avg: {epoch_loss/(b+1)}, acc_now: {n_correct / n_total}')

            # Save average loss per batch
            loss_avg = epoch_loss / (b+1)
            train_acc = n_correct / n_total
            train_stats["train_loss"].append(loss_avg)
            train_stats["train_acc"].append(train_acc)
            logger.info(f"Train loss: {loss_avg:.6f}, train accuracy: {train_acc:.6f}")

            # Save after every epoch
            epoch_save_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            model.save_pretrained(epoch_save_dir)
            tokenizer.save_pretrained(epoch_save_dir)

            # Evaluate every --eval-every epochs
            if epoch % args.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    n_correct = 0
                    n_total = 0
                    for b, batch in enumerate(tqdm(val_loader, desc="Val Batch", ncols=0, total=len(val_loader))):
                        input_ids, attention_mask, labels = [b.to(device) for _,b in batch.items()]
                        outputs = model(input_ids, attention_mask, labels=labels)
                        total_loss += outputs.loss.item()

                        n_correct += torch.eq(outputs.logits.detach().max(dim=1).indices, labels).sum().cpu().item()
                        n_total += labels.size(0)

                    val_loss_avg = total_loss / (b+1)
                    val_acc = n_correct / n_total
                    train_stats["val_loss"].append(val_loss_avg)
                    train_stats["val_acc"].append(val_acc)

                    wandb.log({'train_loss': loss_avg, 'train_acc': train_acc, 'val_loss': val_loss_avg, 'val_acc': val_acc, 'epoch': epoch})
                    logger.info(f"Validation loss: {val_loss_avg:.6f}, validation accuracy: {val_acc:.6f}")

                with open(f"{epoch_save_dir}/train_stats.pkl", "wb") as f:
                    pickle.dump(train_stats, f)

        logger.info(f"Done training model")

    # Evaluate model on test set
    if args.test_eval:
        logger.info(f"Evaluate model on test set")

        labels_list = []
        predicted_list = []
        model.eval()
        with torch.no_grad():
            total_loss = 0
            n_correct = 0
            n_total = 0
            for b, batch in enumerate(tqdm(test_loader, desc="Test Batch", ncols=0, disable=not args.debug, total=len(test_loader))):
                input_ids, attention_mask, labels = [b.to(device) for _,b in batch.items()]
                outputs = model(input_ids, attention_mask, labels=labels)
                total_loss += outputs.loss.item()

                predicted = outputs.logits.detach().max(dim=1).indices
                n_correct += torch.eq(predicted, labels).sum().cpu().item()
                n_total += labels.size(0)
                labels_list += labels.detach().tolist()
                predicted_list += predicted.detach().tolist()

            test_loss_avg = total_loss / (b+1)
            test_acc = n_correct / n_total
            logger.info(f"Test loss: {test_loss_avg:.6f}, test accuracy: {test_acc:.6f}")

            from sklearn.metrics import confusion_matrix, f1_score
            print(f'Confusion matrix:')
            print(confusion_matrix(labels_list, predicted_list, normalize='true'))

            for score_type in ['micro', 'macro', 'weighted']:
                score = f1_score(labels_list, predicted_list, average=score_type)
                print(f'{score_type}-F1: {score}')
    wandb.finish()

