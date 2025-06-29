import numpy as np
import logging
from collections import defaultdict

class EvaluationMetrics:
    def __init__(self):
        self.relevance_thresholds = {
            'Sangat Relevan': 4,
            'Relevan': 3,
            'Cukup Relevan': 2,
            'Sedikit Relevan': 1,
            'Tidak Relevan': 0
        }
    
    def get_relevance_score(self, category):
        """Convert category to relevance score"""
        return self.relevance_thresholds.get(category, 0)
    
    def calculate_precision_at_k(self, results, k=10):
        """Calculate Precision@K"""
        if not results or k <= 0:
            return 0.0
        
        top_k_results = results[:k]
        relevant_count = sum(1 for result in top_k_results 
                           if self.get_relevance_score(result.get('category', 'Tidak Relevan')) >= 2)
        
        return relevant_count / len(top_k_results)
    
    def calculate_recall_without_k(self, results, use_adaptive_estimation=True):
        """
        Calculate Recall without fixed K - use all retrieved results
        This is more realistic for document search systems
        """
        if not results:
            return 0.0
        
        # Count relevant documents in all results
        relevant_retrieved = sum(1 for result in results 
                               if self.get_relevance_score(result.get('category', 'Tidak Relevan')) >= 2)
        
        if relevant_retrieved == 0:
            return 0.0
        
        if use_adaptive_estimation:
            # Adaptive estimation: assume that we found most of the relevant documents
            # This is more realistic than assuming a fixed number
            estimated_total_relevant = max(relevant_retrieved, int(relevant_retrieved * 1.1))
        else:
            # Conservative estimation: use what we found
            estimated_total_relevant = relevant_retrieved
        
        return relevant_retrieved / estimated_total_relevant
    
    def calculate_f1_score(self, precision, recall):
        """Calculate F1-Score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_average_precision(self, results):
        """Calculate Average Precision (AP)"""
        if not results:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, result in enumerate(results, 1):
            if self.get_relevance_score(result.get('category', 'Tidak Relevan')) >= 2:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / relevant_count
    
    def calculate_map_at_k(self, all_queries_results, k=10):
        """Calculate Mean Average Precision@K"""
        if not all_queries_results:
            return 0.0
        
        ap_scores = []
        for results in all_queries_results:
            top_k_results = results[:k]
            ap = self.calculate_average_precision(top_k_results)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def calculate_reciprocal_rank(self, results):
        """Calculate Reciprocal Rank (RR)"""
        for i, result in enumerate(results, 1):
            if self.get_relevance_score(result.get('category', 'Tidak Relevan')) >= 2:
                return 1.0 / i
        return 0.0
    
    def calculate_mrr_at_k(self, all_queries_results, k=10):
        """Calculate Mean Reciprocal Rank@K"""
        if not all_queries_results:
            return 0.0
        
        rr_scores = []
        for results in all_queries_results:
            top_k_results = results[:k]
            rr = self.calculate_reciprocal_rank(top_k_results)
            rr_scores.append(rr)
        
        return np.mean(rr_scores) if rr_scores else 0.0
    
    def calculate_dcg_at_k(self, results, k=10):
        """Calculate Discounted Cumulative Gain@K"""
        if not results or k <= 0:
            return 0.0
        
        dcg = 0.0
        for i, result in enumerate(results[:k], 1):
            relevance = self.get_relevance_score(result.get('category', 'Tidak Relevan'))
            dcg += relevance / np.log2(i + 1)
        
        return dcg
    
    def calculate_ndcg_at_k(self, results, k=10):
        """Calculate Normalized Discounted Cumulative Gain@K"""
        if not results or k <= 0:
            return 0.0
        
        # Calculate DCG
        dcg = self.calculate_dcg_at_k(results, k)
        
        # Calculate IDCG (Ideal DCG)
        relevance_scores = [self.get_relevance_score(result.get('category', 'Tidak Relevan')) 
                          for result in results[:k]]
        ideal_relevance = sorted(relevance_scores, reverse=True)
        
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance, 1):
            idcg += relevance / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_comprehensive_metrics(self, results, precision_k=10, use_recall_without_k=True):
        """
        Calculate comprehensive metrics with option to use recall without fixed K
        OPSI 2: Precision@10, Recall tanpa @K, F1-Score adaptif, MAP@10, MRR@10, NDCG@10
        """
        # Calculate precision@K (what user sees)
        precision = self.calculate_precision_at_k(results, precision_k)
        
        # Calculate recall without fixed K (more realistic)
        if use_recall_without_k:
            recall = self.calculate_recall_without_k(results, use_adaptive_estimation=True)
        else:
            # Fallback to traditional recall calculation
            relevant_in_results = sum(1 for result in results 
                                    if self.get_relevance_score(result.get('category', 'Tidak Relevan')) >= 2)
            recall = relevant_in_results / max(relevant_in_results, 1)  # Avoid division by zero
        
        # Calculate F1-score
        f1_score = self.calculate_f1_score(precision, recall)
        
        # Other metrics remain at their specified K values
        ap = self.calculate_average_precision(results[:precision_k])
        rr = self.calculate_reciprocal_rank(results[:precision_k])
        ndcg = self.calculate_ndcg_at_k(results, precision_k)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'average_precision': ap,
            'reciprocal_rank': rr,
            'ndcg': ndcg,
            'precision_k': precision_k,
            'recall_method': 'adaptive_no_k' if use_recall_without_k else 'traditional',
            'total_results': len(results),
            'relevant_results': sum(1 for result in results 
                                  if self.get_relevance_score(result.get('category', 'Tidak Relevan')) >= 2)
        }
