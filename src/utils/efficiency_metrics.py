"""
Efficiency Metrics Calculator for FFFGNN Models

Provides comprehensive computational complexity analysis including:
- Parameter counts (total, trainable, GNN vs fairness)
- FLOPS estimation for different operations
- Memory usage analysis
- Training/inference time measurement
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class EfficiencyAnalyzer:
    """Comprehensive efficiency analysis for GNN models"""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        self.flop_counts = defaultdict(int)
        self.param_counts = defaultdict(int)
        self.memory_usage = []
        self.timing_data = {}
    
    def analyze_model_complexity(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Analyze complete model complexity
        
        Returns detailed breakdown of:
        - Parameter counts by component
        - FLOPS estimates
        - Memory requirements
        """
        complexity_metrics = {}
        
        # Basic parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Component-wise parameter analysis
        component_params = self._analyze_component_parameters(model)
        
        # FLOPS estimation
        flops_estimate = self._estimate_model_flops(model, component_params)
        
        # Extract total flops from the dictionary
        if isinstance(flops_estimate, dict) and 'total_flops' in flops_estimate:
            total_flops = flops_estimate['total_flops']
        else:
            total_flops = 0
        
        complexity_metrics.update({
            'total_params': total_params,
            'trainable_params': trainable_params,
            'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0,
            'component_breakdown': component_params,
            'flops_estimate': flops_estimate,
            'flops_per_param': total_flops / total_params if total_params > 0 else 0
        })
        
        return complexity_metrics
    
    def _analyze_component_parameters(self, model: torch.nn.Module) -> Dict[str, int]:
        """Analyze parameters by model component"""
        component_params = {
            'gnn_backbone': 0,
            'fairness_monitor': 0,
            'fairness_intervention': 0,
            'other': 0
        }
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            
            # Categorize by component name
            if any(key in name.lower() for key in ['gnn', 'layer', 'conv', 'backbone']):
                component_params['gnn_backbone'] += param_count
            elif any(key in name.lower() for key in ['monitor', 'risk', 'detector']):
                component_params['fairness_monitor'] += param_count
            elif any(key in name.lower() for key in ['intervention', 'projection', 'transform']):
                component_params['fairness_intervention'] += param_count
            else:
                component_params['other'] += param_count
        
        return component_params
    
    def _estimate_model_flops(self, model: torch.nn.Module, component_params: Dict[str, int]) -> int:
        """
        Estimate FLOPS for different model operations
        
        Based on standard GNN operation complexity:
        - Graph convolution: O(E × F_in × F_out) where E = edges, F = features
        - Linear layers: O(N × F_in × F_out) where N = nodes
        - Attention (GAT): O(E × F × heads)
        - Fairness operations: O(N × F) for basic operations
        """
        
        # These are rough estimates based on typical GNN architectures
        # Actual FLOPS depend on specific graph structure and implementation
        
        gnn_flops = self._estimate_gnn_flops(component_params['gnn_backbone'])
        monitor_flops = self._estimate_monitor_flops(component_params['fairness_monitor'])
        intervention_flops = self._estimate_intervention_flops(component_params['fairness_intervention'])
        
        total_flops = gnn_flops + monitor_flops + intervention_flops
        
        return {
            'total_flops': total_flops,
            'gnn_flops': gnn_flops,
            'monitor_flops': monitor_flops,
            'intervention_flops': intervention_flops,
            'flops_breakdown': {
                'gnn_percentage': (gnn_flops / total_flops * 100) if total_flops > 0 else 0,
                'monitor_percentage': (monitor_flops / total_flops * 100) if total_flops > 0 else 0,
                'intervention_percentage': (intervention_flops / total_flops * 100) if total_flops > 0 else 0
            }
        }
    
    def _estimate_gnn_flops(self, gnn_params: int) -> int:
        """Estimate FLOPS for GNN operations"""
        # Rough estimate: each parameter contributes ~100-1000 FLOPS per forward pass
        # depending on graph connectivity and message passing
        base_multiplier = 500  # Conservative estimate
        return gnn_params * base_multiplier
    
    def _estimate_monitor_flops(self, monitor_params: int) -> int:
        """Estimate FLOPS for fairness monitoring"""
        # Monitoring typically involves:
        # - Group statistics computation: O(N)
        # - Similarity/distance calculations: O(N × F)
        # - Threshold comparisons: O(1)
        base_multiplier = 200  # Lower than GNN operations
        return monitor_params * base_multiplier
    
    def _estimate_intervention_flops(self, intervention_params: int) -> int:
        """Estimate FLOPS for fairness intervention"""
        # Intervention typically involves:
        # - Linear projections: O(N × F_in × F_out)
        # - Feature transformations: O(N × F)
        base_multiplier = 300  # Medium complexity
        return intervention_params * base_multiplier
    
    def measure_inference_efficiency(self, model: torch.nn.Module, data_dict: Dict[str, Any], 
                                   num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """
        Measure actual inference efficiency
        
        Args:
            model: The model to measure
            data_dict: Dictionary containing graph, features, sens, labels
            num_runs: Number of inference runs to average
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with timing metrics
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Prepare data
        graph = data_dict['graph']
        features = data_dict['features'] 
        sens = data_dict['sens']
        labels = data_dict['labels']
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(graph, features, sens, labels)
        
        # Synchronize CUDA if available
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure inference time
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                output, fairness_loss = model(graph, features, sens, labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate metrics
        avg_inference_time = total_time / num_runs
        throughput = features.shape[0] * num_runs / total_time  # nodes per second
        
        # Memory measurement (if CUDA available)
        peak_memory_mb = 0
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Single forward pass to measure memory
            with torch.no_grad():
                _ = model(graph, features, sens, labels)
            
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.empty_cache()
        
        return {
            'avg_inference_time': avg_inference_time,
            'inference_time_ms': avg_inference_time * 1000,
            'throughput_nodes_per_sec': throughput,
            'peak_memory_mb': peak_memory_mb,
            'num_runs': num_runs,
            'warmup_runs': warmup_runs
        }
    
    def measure_training_efficiency(self, model: torch.nn.Module, trainer, 
                                  num_epochs: int = 10) -> Dict[str, float]:
        """
        Measure training efficiency metrics
        
        Args:
            model: The model being trained
            trainer: Training instance
            num_epochs: Number of epochs to measure
            
        Returns:
            Dictionary with training efficiency metrics
        """
        start_time = time.perf_counter()
        
        # Simulate training for specified epochs
        initial_epoch = trainer.epoch if hasattr(trainer, 'epoch') else 0
        
        for epoch in range(num_epochs):
            # Single training step
            trainer.train_epoch()
            
            # Memory cleanup every few epochs
            if epoch % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate efficiency metrics
        time_per_epoch = total_time / num_epochs
        
        # Estimate convergence metrics (simplified)
        convergence_speed = self._estimate_convergence_speed(model, trainer)
        
        return {
            'training_time_total': total_time,
            'time_per_epoch': time_per_epoch,
            'epochs_per_hour': 3600 / time_per_epoch if time_per_epoch > 0 else 0,
            'convergence_speed': convergence_speed,
            'num_epochs_measured': num_epochs
        }
    
    def _estimate_convergence_speed(self, model: torch.nn.Module, trainer) -> float:
        """
        Estimate convergence speed based on loss reduction
        This is a simplified heuristic
        """
        # This would need access to training loss history
        # For now, return a placeholder that can be updated with actual logic
        return 1.0
    
    def compare_models_efficiency(self, model_results: list) -> Dict[str, Any]:
        """
        Compare efficiency across multiple models
        
        Args:
            model_results: List of dictionaries with model efficiency metrics
            
        Returns:
            Comparative analysis
        """
        if not model_results:
            return {}
        
        df = pd.DataFrame(model_results)
        
        comparison = {
            'fastest_inference': df.loc[df['inference_time_ms'].idxmin()],
            'most_efficient_memory': df.loc[df['peak_memory_mb'].idxmin()],
            'smallest_model': df.loc[df['total_params'].idxmin()],
            'fastest_training': df.loc[df['time_per_epoch'].idxmin()],
            'best_throughput': df.loc[df['throughput_nodes_per_sec'].idxmax()]
        }
        
        # Calculate efficiency scores (normalized)
        metrics_to_normalize = ['inference_time_ms', 'total_params', 'peak_memory_mb', 'time_per_epoch']
        
        for metric in metrics_to_normalize:
            if metric in df.columns:
                # Lower is better for these metrics
                df[f'{metric}_normalized'] = 1.0 / (df[metric] + 1e-8)
                df[f'{metric}_normalized'] = df[f'{metric}_normalized'] / df[f'{metric}_normalized'].max()
        
        # Combined efficiency score
        efficiency_cols = [col for col in df.columns if col.endswith('_normalized')]
        if efficiency_cols:
            df['efficiency_score'] = df[efficiency_cols].mean(axis=1)
            comparison['most_efficient_overall'] = df.loc[df['efficiency_score'].idxmax()]
        
        return comparison
    
    def generate_efficiency_report(self, model_name: str, metrics: Dict[str, Any]) -> str:
        """Generate human-readable efficiency report"""
        
        report = f"""
Efficiency Analysis Report: {model_name}
{'='*50}

Parameter Analysis:
  - Total Parameters: {metrics.get('total_params', 0):,}
  - Trainable Parameters: {metrics.get('trainable_params', 0):,}
  - Parameter Efficiency: {metrics.get('parameter_efficiency', 0):.3f}
  
Component Breakdown:
"""
        
        if 'component_breakdown' in metrics:
            breakdown = metrics['component_breakdown']
            for component, count in breakdown.items():
                percentage = (count / metrics.get('total_params', 1)) * 100
                report += f"  - {component}: {count:,} ({percentage:.1f}%)\n"
        
        report += f"""
FLOPS Analysis:
  - Total FLOPS: {metrics.get('flops_estimate', {}).get('total_flops', 0):,.0f}
  - FLOPS per Parameter: {metrics.get('flops_per_param', 0):,.0f}
  
FLOPS Breakdown:
"""
        
        if 'flops_estimate' in metrics:
            flops = metrics['flops_estimate']
            if 'flops_breakdown' in flops:
                breakdown = flops['flops_breakdown']
                for component, percentage in breakdown.items():
                    report += f"  - {component}: {percentage:.1f}%\n"
        
        report += f"""
Efficiency Classification: {self._classify_efficiency(metrics)}
        """
        
        return report
    
    def _classify_efficiency(self, metrics: Dict[str, Any]) -> str:
        """Classify model efficiency level"""
        total_params = metrics.get('total_params', 0)
        inference_time = metrics.get('inference_time_ms', 0)
        
        if total_params < 100000 and inference_time < 10:
            return "EXCELLENT - Ultra-lightweight"
        elif total_params < 500000 and inference_time < 50:
            return "GOOD - Lightweight"
        elif total_params < 2000000 and inference_time < 200:
            return "MODERATE - Standard"
        elif total_params < 10000000 and inference_time < 500:
            return "HIGH - Heavy"
        else:
            return "VERY HIGH - Resource intensive"


# Utility functions for easy integration
def quick_efficiency_analysis(model: torch.nn.Module, data_dict: Dict[str, Any], 
                            device: torch.device) -> Dict[str, Any]:
    """
    Quick efficiency analysis wrapper
    
    Returns comprehensive efficiency metrics in one call
    """
    analyzer = EfficiencyAnalyzer()
    
    # Model complexity analysis
    complexity = analyzer.analyze_model_complexity(model)
    
    # Inference efficiency (if data provided)
    if data_dict and all(key in data_dict for key in ['graph', 'features', 'sens', 'labels']):
        inference = analyzer.measure_inference_efficiency(model, data_dict)
        complexity.update(inference)
    
    # Generate report
    model_name = model.__class__.__name__
    report = analyzer.generate_efficiency_report(model_name, complexity)
    
    complexity['efficiency_report'] = report
    
    return complexity


# For backward compatibility and easy import
__all__ = ['EfficiencyAnalyzer', 'quick_efficiency_analysis']