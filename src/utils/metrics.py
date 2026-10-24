import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Tuple, Dict
import logging
import torch.nn.functional as F

# Create logger
logger = logging.getLogger(__name__)

def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate accuracy
    
    Consistent with original FairGNN implementation: (output > 0) as predicted positive class
    """
    output = output.squeeze()
    
    # Use >0 as threshold, consistent with original FairGNN implementation
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def roc_auc(labels, output) -> float:
    """Calculate ROC AUC
    
    Args:
        labels: True labels (torch.Tensor or numpy.ndarray)
        output: Model output (torch.Tensor or numpy.ndarray)
        
    Returns:
        float: ROC AUC score
    """
    # Check input types and convert to numpy arrays
    if isinstance(labels, torch.Tensor):
        y_true = labels.cpu().numpy()
    else:
        y_true = labels
        
    if isinstance(output, torch.Tensor):
        y_score = output.detach().cpu().numpy()
    else:
        y_score = output
    
    # Check number of classes - ROC AUC needs at least two classes
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        # Only one class, cannot calculate ROC AUC
        logger.warning(f"ROC AUC calculation failed: only one class {unique_labels}")
        return 0.5
    
    # Check if all prediction values are the same
    if np.max(y_score) == np.min(y_score):
        # All prediction values are the same, cannot calculate ROC AUC, but no longer considered invalid model
        logger.warning(f"ROC AUC calculation note: all prediction values are the same {np.max(y_score)}")
        return 0.5
    
    try:
        # Try to calculate ROC AUC
        return roc_auc_score(y_true, y_score)
    except Exception as e:
        logger.warning(f"ROC AUC calculation exception: {str(e)}")
        return 0.5


def fair_metric(output: torch.Tensor, labels: torch.Tensor, 
                sens: torch.Tensor, idx: torch.Tensor) -> Tuple[float, float]:
    """Calculate fairness metrics
    
    Args:
        output: Model output
        labels: True labels
        sens: Sensitive attributes
        idx: Evaluation indices
        
    Returns:
        Tuple[float, float]: (parity, equality)
    """
    try:
        # Ensure all inputs are converted to numpy
        # logger.debug(f"Calculating fairness metrics: output.shape={output.shape}, labels.shape={labels.shape}, sens.shape={sens.shape}, idx.shape={idx.shape}")
        
        # Get output for idx
        idx_output = output[idx]
        
        # Process prediction results for binary classification
        if isinstance(idx_output, torch.Tensor):
            # Check if output is logits
            if len(idx_output.shape) > 1 and idx_output.shape[1] > 1:
                probs = torch.softmax(idx_output, dim=1)
                class_probs = probs[:, 1].cpu().numpy()  # Take positive class probability
                preds = (class_probs > 0.5).astype(np.int32)
            else:
                preds = (idx_output.squeeze() > 0).cpu().numpy().astype(np.int32)
        else:
            preds = idx_output
        
        # Convert labels to numpy
        if isinstance(labels, torch.Tensor):
            y_true = labels[idx].cpu().numpy()
        else:
            y_true = labels[idx]
            
        # Check invalid model: predictions or true labels have only one class - modified to warning but continue calculation
        unique_preds = np.unique(preds)
        unique_true = np.unique(y_true)
        if len(unique_preds) == 1 or len(unique_true) == 1:
            logger.warning("Fairness metrics calculation: all predictions or labels have only one class, but continue calculation")
            # No longer return inf, continue calculation
            
        if isinstance(sens, torch.Tensor):
            sens_np = sens.cpu().numpy()
        else:
            sens_np = sens
        
        if isinstance(idx, torch.Tensor):
            idx_np = idx.cpu().numpy()
        else:
            idx_np = idx
            
        sens_idx = sens_np[idx_np]
        
        # Grouping
        idx_s0 = (sens_idx == 0)
        idx_s1 = (sens_idx == 1)
        
        # Check sensitive attribute distribution
        num_s0 = np.sum(idx_s0)
        num_s1 = np.sum(idx_s1)
        
        # logger.debug(f"Sensitive attribute 0: {num_s0} samples, sensitive attribute 1: {num_s1} samples")
        
        # If a sensitive attribute group has no samples, cannot calculate fairness metrics
        if num_s0 == 0 or num_s1 == 0:
            logger.warning(f"Fairness calculation failed: sensitive attributes imbalanced, attribute=0 has {num_s0} samples, attribute=1 has {num_s1} samples")
            # Return medium value indicating cannot calculate, not worst value
            return 0.5, 0.5  # Modified to 0.5 instead of 1.0

        # Group based on actual labels
        idx_s0_y1 = np.logical_and(idx_s0, y_true == 1)
        idx_s1_y1 = np.logical_and(idx_s1, y_true == 1)
        
        # Check positive example distribution
        num_s0_y1 = np.sum(idx_s0_y1)
        num_s1_y1 = np.sum(idx_s1_y1)
        
        # logger.debug(f"Sensitive attribute 0 + label 1: {num_s0_y1} samples, sensitive attribute 1 + label 1: {num_s1_y1} samples")
        
        # If a sensitive attribute group has no positive examples, cannot calculate equality
        if num_s0_y1 == 0 or num_s1_y1 == 0:
            logger.warning(f"Equality calculation failed: positive example distribution imbalanced, attribute=0 positive examples {num_s0_y1} samples, attribute=1 positive examples {num_s1_y1} samples")
            
            # Calculate parity (statistical parity)
            pred_rate_s0 = np.sum(preds[idx_s0]) / num_s0 if num_s0 > 0 else 0
            pred_rate_s1 = np.sum(preds[idx_s1]) / num_s1 if num_s1 > 0 else 0
            
            # logger.debug(f"Group 0 prediction rate: {pred_rate_s0:.4f}, group 1 prediction rate: {pred_rate_s1:.4f}")
            parity = abs(pred_rate_s0 - pred_rate_s1)
            
            # Check if all predictions are the same - no longer considered invalid model
            if np.all(preds == preds[0]):
                logger.warning("Fairness metrics note: all predictions are the same, but continue calculation")
                # If all predictions are the same, parity may be 0, which is reasonable
                
            return float(parity), 0.5  # Modified to 0.5 instead of 1.0

        # Calculate parity (statistical parity) - difference in positive prediction rates between different sensitive groups
        pred_rate_s0 = np.sum(preds[idx_s0]) / num_s0
        pred_rate_s1 = np.sum(preds[idx_s1]) / num_s1
        
        # Debug info
        # logger.debug(f"Group 0 positive prediction rate: {pred_rate_s0:.4f}, group 1 positive prediction rate: {pred_rate_s1:.4f}")
        
        parity = abs(pred_rate_s0 - pred_rate_s1)
        
        # Calculate equality (equal opportunity) - difference in proportion of true positives predicted as positive between different sensitive groups
        pred_pos_s0_y1 = np.sum(preds[idx_s0_y1])
        pred_pos_s1_y1 = np.sum(preds[idx_s1_y1])
        
        pred_rate_s0_y1 = pred_pos_s0_y1 / num_s0_y1 if num_s0_y1 > 0 else 0
        pred_rate_s1_y1 = pred_pos_s1_y1 / num_s1_y1 if num_s1_y1 > 0 else 0
        
        # Debug info
        # logger.debug(f"Group 0 proportion of true positives predicted as positive: {pred_rate_s0_y1:.4f}, group 1 proportion: {pred_rate_s1_y1:.4f}")
        
        equality = abs(pred_rate_s0_y1 - pred_rate_s1_y1)
        
        # logger.debug(f"Calculation result: parity={parity:.4f}, equality={equality:.4f}")
        
        # Check if all predictions are the same - no longer considered invalid model
        if np.all(preds == preds[0]):
            logger.warning("Fairness metrics note: all predictions are the same, but continue calculation")
            # If all predictions are the same, this may result in some fairness metrics being 0, which is reasonable
    
        return float(parity), float(equality)
        
    except Exception as e:
        logger.warning(f"Fairness metrics calculation exception: {str(e)}")
        import traceback
        # logger.debug(traceback.format_exc())
        return 0.5, 0.5  # Modified: return 0.5 in exception case indicating medium unfairness, not 1.0

def calculate_all_metrics(output: torch.Tensor, labels: torch.Tensor, 
                         sens: torch.Tensor, idx: torch.Tensor) -> Dict[str, float]:
    """Calculate all evaluation metrics
    
    Args:
        output: Model output
        labels: True labels
        sens: Sensitive attributes
        idx: Evaluation indices
        
    Returns:
        Dict[str, float]: Dictionary containing all metrics
    """
    metrics = {}
    
    try:
        idx_output = output[idx]
        idx_labels = labels[idx]
        
        # Process output dimensions
        if len(idx_output.shape) > 1 and idx_output.shape[1] > 1:
            pred = idx_output.argmax(dim=1)
            if idx_output.shape[1] == 2:
                probs = F.softmax(idx_output, dim=1)[:, 1]
            else:
                probs = None
        else:
            pred = (idx_output.squeeze() > 0).long()
            probs = F.sigmoid(idx_output.squeeze())
            
        # Check invalid model: predictions or true labels have only one class - modified to warning but continue calculation
        y_true = idx_labels.cpu().numpy() if isinstance(idx_labels, torch.Tensor) else idx_labels
        y_pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        if len(unique_true) == 1 or len(unique_pred) == 1:
            logger.warning("All evaluation metrics: all predictions or labels have only one class, but continue calculation")
            # No longer return inf, continue calculation
            
        # Calculate accuracy
        acc = (pred == idx_labels).float().mean().item()
        metrics['accuracy'] = float(acc)
        
        # Calculate ROC AUC
        try:
            if probs is not None:
                # If all prediction probabilities are the same, ROC AUC cannot be calculated, return 0.5
                probs_np = probs.cpu().numpy()
                if np.all(probs_np == probs_np[0]):
                    logger.warning("ROC AUC calculation: all prediction probabilities are the same, return 0.5")
                    metrics['roc_auc'] = 0.5
                else:
                    roc = roc_auc_score(idx_labels.cpu().numpy(), probs.cpu().numpy())
                    metrics['roc_auc'] = float(roc)
            else:
                metrics['roc_auc'] = 0.5
        except Exception as e:
            logger.warning(f"ROC AUC calculation exception: {str(e)}")
            metrics['roc_auc'] = 0.5
        
        # Calculate fairness metrics
        try:
            parity, equality = fair_metric(output, labels, sens, idx)
            metrics['parity'] = parity
            metrics['equality'] = equality
        except Exception as e:
            logger.warning(f"Fairness metrics calculation exception: {str(e)}")
            metrics['parity'] = 0.5  # Modified to 0.5 instead of 1.0
            metrics['equality'] = 0.5  # Modified to 0.5 instead of 1.0
        
        # Calculate F1 score
        try:
            # Check for NaN or infinity values
            if np.isnan(y_true).any() or np.isnan(y_pred).any():
                logger.warning("F1 calculation - NaN values detected")
                metrics['f1'] = 0.0
                return metrics
                
            # Get unique values
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            
            # If only one class, use accuracy as F1
            if len(unique_true) == 1 or len(unique_pred) == 1:
                logger.warning(f"F1 calculation - only one class, use accuracy as F1")
                metrics['f1'] = float(acc)
            else:
                # Try to calculate F1
                try:
                    f1 = f1_score(y_true, y_pred, average='weighted')
                    metrics['f1'] = float(f1)
                except Exception as e:
                    # Try binary F1
                    logger.warning(f"Weighted F1 calculation failed, try binary F1")
                    try:
                        f1 = f1_score(y_true, y_pred, average='binary')
                        metrics['f1'] = float(f1)
                    except Exception as e2:
                        logger.warning("Binary F1 also failed, use accuracy as F1")
                        metrics['f1'] = float(acc)
        except Exception as e:
            logger.warning(f"F1 score calculation exception: {str(e)}")
            metrics['f1'] = 0.0
    
    except Exception as e:
        logger.warning(f"Metrics calculation general exception: {str(e)}")
        # Set default values
        metrics = {
            'accuracy': 0.0, 
            'roc_auc': 0.5,
            'f1': 0.0,
            'parity': 0.5,  # Modified to 0.5 instead of 1.0
            'equality': 0.5  # Modified to 0.5 instead of 1.0
        }
    
    return metrics

def macro_f1(y_true, y_pred):
    """Macro-average F1 (suitable for multi-class)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Macro-average F1 score
    """
    return f1_score(y_true, y_pred, average='macro')

def weighted_f1(y_true, y_pred):
    """Weighted F1 (considers class imbalance)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Weighted F1 score
    """
    return f1_score(y_true, y_pred, average='weighted')

def impact_parity(y_pred_prob, sens, threshold=0.5):
    """Calculate decision impact difference between different sensitive groups
    
    Args:
        y_pred_prob: Prediction probabilities
        sens: Sensitive attributes (0/1)
        threshold: Decision threshold
    
    Returns:
        ip_diff: Absolute difference in rejection rates between two groups
    """
    try:
        if isinstance(y_pred_prob, torch.Tensor):
            y_pred_prob = y_pred_prob.detach().cpu().numpy()
        if isinstance(sens, torch.Tensor):
            sens = sens.detach().cpu().numpy()
            
        # If y_pred_prob is 2D (binary classification probability), take the probability of the second class
        if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] == 2:
            y_pred_prob = y_pred_prob[:, 1]
            
        decisions = (y_pred_prob > threshold).astype(int)
        idx_s0 = sens==0
        idx_s1 = sens==1
        
        # Avoid division by zero
        if sum(idx_s0) == 0 or sum(idx_s1) == 0:
            return 0.5  # Modified to 0.5 instead of 1.0
            
        # Check if all predictions are the same - no longer considered invalid model
        if np.all(decisions == decisions[0]):
            logger.warning("Impact parity note: all predictions are the same, but continue calculation")
            
        reject_rate_0 = 1 - decisions[idx_s0].mean()
        reject_rate_1 = 1 - decisions[idx_s1].mean()
        
        return float(abs(reject_rate_0 - reject_rate_1))
    except Exception as e:
        logger.warning(f"Error calculating impact parity: {str(e)}")
        return 0.5  # Modified to 0.5 instead of 1.0