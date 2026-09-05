"""
Module 0 — Evaluation Metrics for Chargeback Representment & Evidence Automation.

Implements honest currency and performance metrics:
- Count-based win rate vs. Amount-weighted win rate
- Precision-Recall AUC (PR-AUC) and ROC-AUC
- Per-criterion precision, recall, F1, and PR-AUC
- Real-world currency (₹) cost model (FP cost, FN cost, Net recovered revenue)
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


def count_based_win_rate(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Optional[Union[np.ndarray, pd.Series]] = None,
) -> float:
    """
    Compute count-based win rate.
    
    If y_pred is provided, calculates the fraction of predicted wins (contested disputes)
    that actually resulted in a win (y_true == 1 among cases where y_pred == 1).
    If y_pred is None, calculates the baseline win rate over all cases in y_true.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    if y_pred is None:
        if len(y_true_arr) == 0:
            return 0.0
        return float(np.mean(y_true_arr == 1))
    
    y_pred_arr = np.asarray(y_pred, dtype=int)
    contested_mask = (y_pred_arr == 1)
    if not np.any(contested_mask):
        return 0.0
    return float(np.mean(y_true_arr[contested_mask] == 1))


def amount_weighted_win_rate(
    y_true: Union[np.ndarray, pd.Series],
    amounts: Union[np.ndarray, pd.Series],
    y_pred: Optional[Union[np.ndarray, pd.Series]] = None,
) -> float:
    """
    Compute amount-weighted win rate.
    
    Amount-weighted win rate = (Sum of amounts of won contested disputes) / 
                               (Sum of amounts of all contested disputes)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    amt_arr = np.asarray(amounts, dtype=float)
    
    if y_pred is None:
        total_amt = np.sum(amt_arr)
        if total_amt <= 0:
            return 0.0
        won_amt = np.sum(amt_arr[y_true_arr == 1])
        return float(won_amt / total_amt)
    
    y_pred_arr = np.asarray(y_pred, dtype=int)
    contested_mask = (y_pred_arr == 1)
    contested_amt = np.sum(amt_arr[contested_mask])
    if contested_amt <= 0:
        return 0.0
    
    won_contested_amt = np.sum(amt_arr[contested_mask & (y_true_arr == 1)])
    return float(won_contested_amt / contested_amt)


def compute_pr_auc(
    y_true: Union[np.ndarray, pd.Series],
    y_score: Union[np.ndarray, pd.Series],
) -> float:
    """
    Compute Area Under the Precision-Recall Curve (PR-AUC) using average precision.
    Handles edge cases with single class safely.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    
    if len(np.unique(y_true_arr)) < 2:
        return 0.0
    
    return float(average_precision_score(y_true_arr, y_score_arr))


def compute_roc_auc(
    y_true: Union[np.ndarray, pd.Series],
    y_score: Union[np.ndarray, pd.Series],
) -> float:
    """
    Compute Area Under ROC Curve (ROC-AUC).
    Handles edge cases with single class safely.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    
    if len(np.unique(y_true_arr)) < 2:
        return 0.5
    
    return float(roc_auc_score(y_true_arr, y_score_arr))


def compute_financial_pnl(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    amounts: Union[np.ndarray, pd.Series],
    review_cost: float = 15.0,
    fight_and_lose_fee: float = 25.0,
) -> Dict[str, float]:
    """
    Compute financial profit/loss and cost metrics in Currency (₹).
    
    Cost model:
    - True Positive (Contested & Won): Recovers (Dispute Amount - review_cost)
    - False Positive (Contested & Lost): Loses (review_cost + fight_and_lose_fee)
    - False Negative (Not Contested & Was Winnable): Opportunity loss (Dispute Amount)
    - True Negative (Not Contested & Was Unwinnable): Avoided fees (Saved review_cost + fight_and_lose_fee)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    amt_arr = np.asarray(amounts, dtype=float)
    
    tp_mask = (y_pred_arr == 1) & (y_true_arr == 1)
    fp_mask = (y_pred_arr == 1) & (y_true_arr == 0)
    fn_mask = (y_pred_arr == 0) & (y_true_arr == 1)
    tn_mask = (y_pred_arr == 0) & (y_true_arr == 0)
    
    recovered_revenue = float(np.sum(amt_arr[tp_mask]))
    total_review_cost = float(np.sum(y_pred_arr == 1) * review_cost)
    total_fight_lose_fees = float(np.sum(fp_mask) * fight_and_lose_fee)
    
    net_pnl = recovered_revenue - total_review_cost - total_fight_lose_fees
    
    fp_cost_incurred = float(np.sum(fp_mask) * (review_cost + fight_and_lose_fee))
    fn_opportunity_loss = float(np.sum(amt_arr[fn_mask]))
    
    potential_total_winnable_amt = float(np.sum(amt_arr[y_true_arr == 1]))
    recovery_rate = (recovered_revenue / potential_total_winnable_amt) if potential_total_winnable_amt > 0 else 0.0
    
    return {
        "recovered_revenue_inr": recovered_revenue,
        "net_pnl_inr": net_pnl,
        "review_costs_inr": total_review_cost,
        "fight_lose_fees_inr": total_fight_lose_fees,
        "fp_cost_incurred_inr": fp_cost_incurred,
        "fn_opportunity_loss_inr": fn_opportunity_loss,
        "recovery_rate": recovery_rate,
        "count_contested": int(np.sum(y_pred_arr == 1)),
        "count_won": int(np.sum(tp_mask)),
        "count_lost": int(np.sum(fp_mask)),
        "count_missed": int(np.sum(fn_mask)),
    }


def evaluate_dispute_model(
    y_true: Union[np.ndarray, pd.Series],
    y_score: Union[np.ndarray, pd.Series],
    amounts: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
    review_cost: float = 15.0,
    fight_and_lose_fee: float = 25.0,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a dispute outcome prediction model.
    Reports count-based win rate, amount-weighted win rate, PR-AUC, ROC-AUC,
    Brier score, and honest currency metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred_arr = (y_score_arr >= threshold).astype(int)
    amt_arr = np.asarray(amounts, dtype=float)
    
    pr_auc = compute_pr_auc(y_true_arr, y_score_arr)
    roc_auc = compute_roc_auc(y_true_arr, y_score_arr)
    brier = float(brier_score_loss(y_true_arr, y_score_arr))
    
    cnt_win_rate = count_based_win_rate(y_true_arr, y_pred_arr)
    amt_win_rate = amount_weighted_win_rate(y_true_arr, amt_arr, y_pred_arr)
    
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    
    financials = compute_financial_pnl(
        y_true_arr,
        y_pred_arr,
        amt_arr,
        review_cost=review_cost,
        fight_and_lose_fee=fight_and_lose_fee,
    )
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "count_win_rate": cnt_win_rate,
        "amount_weighted_win_rate": amt_win_rate,
        "threshold": threshold,
        "financials": financials,
    }


def evaluate_criterion_breakdown(
    y_true_dict: Dict[str, Union[np.ndarray, pd.Series]],
    y_score_dict: Dict[str, Union[np.ndarray, pd.Series]],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluates individual rubric criterion predictions against ground truth labels.
    Returns per-criterion Precision, Recall, F1, PR-AUC, and ROC-AUC.
    """
    rows = []
    for crit_name, y_true in y_true_dict.items():
        if crit_name not in y_score_dict:
            continue
        y_true_arr = np.asarray(y_true, dtype=int)
        y_score_arr = np.asarray(y_score_dict[crit_name], dtype=float)
        y_pred_arr = (y_score_arr >= threshold).astype(int)
        
        pr_auc = compute_pr_auc(y_true_arr, y_score_arr)
        roc_auc = compute_roc_auc(y_true_arr, y_score_arr)
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        support = int(np.sum(y_true_arr == 1))
        
        rows.append({
            "criterion": crit_name,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "positive_support": support,
            "total_samples": len(y_true_arr),
            "match_rate": float(np.mean(y_true_arr == 1)),
        })
        
    return pd.DataFrame(rows)
