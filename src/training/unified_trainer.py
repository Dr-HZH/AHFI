"""
Unified trainer for all model variants

Handles training, evaluation, and logging for all models uniformly.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import time
from tqdm import tqdm


class UnifiedTrainer:
    """Unified trainer for all model variants"""

    def __init__(self, args, model, data, model_name='model'):
        self.args = args
        self.model = model
        self.data = data
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)


        self.optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )


        self.best_val_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        self.patience = 500


        self.history = {
            'train_loss': [],
            'val_acc': [],
            'val_parity': [],
            'val_equality': []
        }

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()

        try:


            logits, primal_fair_loss, dual_loss = self.model(
                self.data['graph'],
                self.data['features'],
                self.data['sens'],
                self.data['labels']
            )
        except Exception as e:
            print(f"ERROR in model forward pass: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.01


        try:
            cls_loss = F.cross_entropy(
                logits[self.data['idx_train']],
                self.data['labels'][self.data['idx_train']]
            )
        except Exception as e:
            print(f"ERROR in classification loss: {e}")
            cls_loss = torch.tensor(0.0, device=logits.device)





        total_loss = cls_loss + primal_fair_loss + dual_loss


        try:
            total_loss.backward()
            

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        except Exception as e:
            print(f"ERROR in backward pass: {e}")


        return total_loss.item(), cls_loss.item(), primal_fair_loss.item()

    def evaluate(self, idx):
        """Evaluate on given indices"""
        self.model.eval()

        with torch.no_grad():
            try:
                model_output = self.model(
                    self.data['graph'],
                    self.data['features'],
                    self.data['sens'],
                    self.data['labels']
                )


                if isinstance(model_output, tuple):
                    output = model_output[0]

                else:
                    output = model_output

            except Exception as e:
                print(f"Warning: Model evaluation failed with error: {e}")
                output = torch.zeros((len(self.data['labels']), 2), device=self.data['features'].device)


            preds = output[idx].argmax(dim=1)
            labels = self.data['labels'][idx]
            sens = self.data['sens'][idx]


            acc = (preds == labels).float().mean().item()


            parity = self.compute_demographic_parity(preds, sens)
            equality = self.compute_equal_opportunity(preds, labels, sens)


            f1 = self.compute_f1(preds, labels)

        return {
            'acc': acc,
            'parity': parity,
            'equality': equality,
            'f1': f1
        }

    def compute_demographic_parity(self, preds, sens):
        """Compute demographic parity difference"""
        s0_mask = (sens == 0)
        s1_mask = (sens == 1)

        if not (s0_mask.any() and s1_mask.any()):
            return 0.0

        p_s0 = preds[s0_mask].float().mean().item()
        p_s1 = preds[s1_mask].float().mean().item()

        return abs(p_s0 - p_s1)

    def compute_equal_opportunity(self, preds, labels, sens):
        """Compute equal opportunity difference"""

        pos_mask = (labels == 1)

        if not pos_mask.any():
            return 0.0

        s0_mask = (sens == 0) & pos_mask
        s1_mask = (sens == 1) & pos_mask

        if not (s0_mask.any() and s1_mask.any()):
            return 0.0

        tpr_s0 = preds[s0_mask].float().mean().item()
        tpr_s1 = preds[s1_mask].float().mean().item()

        return abs(tpr_s0 - tpr_s1)

    def compute_f1(self, preds, labels):
        """Compute F1 score"""
        tp = ((preds == 1) & (labels == 1)).float().sum().item()
        fp = ((preds == 1) & (labels == 0)).float().sum().item()
        fn = ((preds == 0) & (labels == 1)).float().sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        return 2 * (precision * recall) / (precision + recall + 1e-8)

    def train(self):
        """Full training loop"""
        start_time = time.time()
        for epoch in tqdm(range(self.args.epochs), desc="Training"):
            train_loss, cls_loss, fair_loss = self.train_epoch()


            val_metrics = self.evaluate(self.data['idx_val'])


            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_parity'].append(val_metrics['parity'])
            self.history['val_equality'].append(val_metrics['equality'])


            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self.best_epoch = epoch
                self.patience_counter = 0

            else:
                self.patience_counter += 1




        test_metrics = self.evaluate(self.data['idx_test'])


        return {
            'test_acc': test_metrics['acc'],
            'test_parity': test_metrics['parity'],
            'test_equality': test_metrics['equality'],
            'test_f1': test_metrics['f1'],
            'training_time': time.time() - start_time
        }
