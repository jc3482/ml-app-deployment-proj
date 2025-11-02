"""
Evaluation metrics for detection and retrieval.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DetectionMetrics:
    """
    Metrics for evaluating ingredient detection performance.
    
    Implements:
    - Precision, Recall, F1 score
    - mAP (mean Average Precision)
    - mAP@IoU thresholds (mAP50, mAP75, etc.)
    """
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Initialize detection metrics.
        
        Args:
            iou_thresholds: List of IoU thresholds for mAP calculation
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75, 0.95]
        
        self.iou_thresholds = iou_thresholds
    
    def calculate_iou(
        self,
        box1: List[float],
        box2: List[float],
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: Bounding box [x1, y1, x2, y2]
            box2: Bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        # Determine intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0.0
        
        return iou
    
    def precision_recall_f1(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (precision, recall, f1)
            
        TODO: Implement matching logic
        - Match predictions to ground truth based on IoU
        - Count true positives, false positives, false negatives
        - Calculate precision, recall, F1
        """
        # TODO: Implement matching and metric calculation
        
        # Placeholder
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        
        return precision, recall, f1
    
    def average_precision(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Calculate Average Precision (AP) for a single class.
        
        Args:
            predictions: List of predicted detections with confidence scores
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            Average Precision score
            
        TODO: Implement AP calculation
        - Sort predictions by confidence
        - Calculate precision-recall curve
        - Compute area under curve (AP)
        """
        # TODO: Implement AP calculation
        
        return 0.0
    
    def mean_average_precision(
        self,
        predictions: Dict[str, List[Dict]],
        ground_truth: Dict[str, List[Dict]],
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Calculate mean Average Precision (mAP) across all classes.
        
        Args:
            predictions: Dictionary mapping class names to predictions
            ground_truth: Dictionary mapping class names to ground truth
            iou_threshold: IoU threshold for matching
            
        Returns:
            mAP score
            
        TODO: Implement mAP calculation
        - Calculate AP for each class
        - Average across all classes
        """
        # TODO: Implement mAP calculation
        
        return 0.0
    
    def evaluate(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> Dict[str, float]:
        """
        Run complete evaluation and return all metrics.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth annotations
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Calculate metrics at different IoU thresholds
        for iou_thresh in self.iou_thresholds:
            precision, recall, f1 = self.precision_recall_f1(
                predictions, ground_truth, iou_thresh
            )
            
            results[f"precision@{iou_thresh}"] = precision
            results[f"recall@{iou_thresh}"] = recall
            results[f"f1@{iou_thresh}"] = f1
        
        # Calculate mAP
        results["mAP50"] = 0.0  # TODO: Implement
        results["mAP75"] = 0.0  # TODO: Implement
        results["mAP"] = 0.0    # TODO: Implement (average over multiple IoU thresholds)
        
        return results


class RetrievalMetrics:
    """
    Metrics for evaluating recipe retrieval performance.
    
    Implements:
    - Recall@K
    - nDCG@K (Normalized Discounted Cumulative Gain)
    - MRR (Mean Reciprocal Rank)
    - MAP (Mean Average Precision)
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize retrieval metrics.
        
        Args:
            k_values: List of k values for Recall@K and nDCG@K
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]
        
        self.k_values = k_values
    
    def recall_at_k(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int,
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved recipe IDs (in order)
            relevant: List of relevant recipe IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        hits = len(retrieved_k.intersection(relevant_set))
        recall = hits / len(relevant_set)
        
        return recall
    
    def dcg_at_k(
        self,
        relevances: List[float],
        k: int,
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at K.
        
        Args:
            relevances: List of relevance scores (in retrieval order)
            k: Number of top results to consider
            
        Returns:
            DCG@K score
        """
        relevances_k = relevances[:k]
        
        dcg = 0.0
        for i, rel in enumerate(relevances_k):
            dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        return dcg
    
    def ndcg_at_k(
        self,
        retrieved: List[int],
        relevant_with_scores: Dict[int, float],
        k: int,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        Args:
            retrieved: List of retrieved recipe IDs (in order)
            relevant_with_scores: Dictionary mapping recipe IDs to relevance scores
            k: Number of top results to consider
            
        Returns:
            nDCG@K score
        """
        # Get relevance scores for retrieved items
        relevances = [relevant_with_scores.get(item_id, 0.0) for item_id in retrieved[:k]]
        
        # Calculate DCG
        dcg = self.dcg_at_k(relevances, k)
        
        # Calculate ideal DCG (sort by relevance)
        ideal_relevances = sorted(relevant_with_scores.values(), reverse=True)
        idcg = self.dcg_at_k(ideal_relevances, k)
        
        # Calculate nDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    
    def mean_reciprocal_rank(
        self,
        retrieved: List[int],
        relevant: List[int],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved: List of retrieved recipe IDs (in order)
            relevant: List of relevant recipe IDs
            
        Returns:
            MRR score
        """
        relevant_set = set(relevant)
        
        for i, item_id in enumerate(retrieved):
            if item_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def average_precision(
        self,
        retrieved: List[int],
        relevant: List[int],
    ) -> float:
        """
        Calculate Average Precision.
        
        Args:
            retrieved: List of retrieved recipe IDs (in order)
            relevant: List of relevant recipe IDs
            
        Returns:
            Average Precision score
        """
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        
        precision_sum = 0.0
        num_relevant = 0
        
        for i, item_id in enumerate(retrieved):
            if item_id in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precision_sum += precision_at_i
        
        ap = precision_sum / len(relevant_set)
        
        return ap
    
    def evaluate(
        self,
        retrieved: List[int],
        relevant: List[int],
        relevant_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        """
        Run complete evaluation and return all metrics.
        
        Args:
            retrieved: List of retrieved recipe IDs
            relevant: List of relevant recipe IDs
            relevant_scores: Optional relevance scores for nDCG
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Recall@K
        for k in self.k_values:
            results[f"recall@{k}"] = self.recall_at_k(retrieved, relevant, k)
        
        # nDCG@K
        if relevant_scores:
            for k in self.k_values:
                results[f"ndcg@{k}"] = self.ndcg_at_k(retrieved, relevant_scores, k)
        
        # MRR
        results["mrr"] = self.mean_reciprocal_rank(retrieved, relevant)
        
        # MAP
        results["map"] = self.average_precision(retrieved, relevant)
        
        return results
    
    def batch_evaluate(
        self,
        batch_retrieved: List[List[int]],
        batch_relevant: List[List[int]],
    ) -> Dict[str, float]:
        """
        Evaluate multiple queries and average results.
        
        Args:
            batch_retrieved: List of retrieved recipe lists
            batch_relevant: List of relevant recipe lists
            
        Returns:
            Dictionary with averaged metrics
        """
        all_results = []
        
        for retrieved, relevant in zip(batch_retrieved, batch_relevant):
            results = self.evaluate(retrieved, relevant)
            all_results.append(results)
        
        # Average all metrics
        averaged_results = {}
        if all_results:
            for key in all_results[0].keys():
                averaged_results[key] = np.mean([r[key] for r in all_results])
        
        return averaged_results

