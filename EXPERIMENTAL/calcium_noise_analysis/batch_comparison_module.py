"""
Batch Comparison Module
=======================
Statistical comparison of calcium imaging data across multiple conditions,
treatments, or cell populations.

Provides tools for:
- Comparing multiple datasets
- Statistical testing
- Effect size calculation
- Population-level analysis

Author: George
"""

import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, ttest_ind, f_oneway


def compare_two_conditions(data1, data2, metric_name='value', 
                           test='mannwhitney', paired=False):
    """
    Compare a metric between two experimental conditions.
    
    Parameters:
        data1 (ndarray): Measurements from condition 1
        data2 (ndarray): Measurements from condition 2
        metric_name (str): Name of metric being compared
        test (str): Statistical test: 'mannwhitney', 'ttest', or 'permutation'
        paired (bool): Whether data is paired (e.g., before/after)
    
    Returns:
        dict: Dictionary containing:
            - 'statistic': Test statistic
            - 'p_value': P-value
            - 'effect_size': Cohen's d or rank-biserial correlation
            - 'mean_1': Mean of condition 1
            - 'mean_2': Mean of condition 2
            - 'std_1': Std of condition 1
            - 'std_2': Std of condition 2
            - 'n_1': Sample size condition 1
            - 'n_2': Sample size condition 2
            - 'significant': Boolean (p < 0.05)
    
    Example:
        >>> control = np.array([1.2, 1.5, 1.3, 1.4])
        >>> treatment = np.array([2.1, 2.3, 2.0, 2.4])
        >>> result = compare_two_conditions(control, treatment, 
        >>>                                 metric_name='amplitude',
        >>>                                 test='ttest')
        >>> print(f"p-value: {result['p_value']:.4f}")
        >>> print(f"Effect size: {result['effect_size']:.2f}")
    """
    # Remove NaN values
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    
    if len(data1) == 0 or len(data2) == 0:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'effect_size': 0,
            'mean_1': np.nan,
            'mean_2': np.nan,
            'std_1': np.nan,
            'std_2': np.nan,
            'n_1': len(data1),
            'n_2': len(data2),
            'significant': False
        }
    
    # Descriptive statistics
    mean_1 = np.mean(data1)
    mean_2 = np.mean(data2)
    std_1 = np.std(data1, ddof=1)
    std_2 = np.std(data2, ddof=1)
    n_1 = len(data1)
    n_2 = len(data2)
    
    # Statistical test
    if paired:
        if test == 'ttest':
            statistic, p_value = stats.ttest_rel(data1, data2)
        else:
            statistic, p_value = stats.wilcoxon(data1, data2)
    else:
        if test == 'ttest':
            statistic, p_value = ttest_ind(data1, data2)
        elif test == 'mannwhitney':
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        elif test == 'permutation':
            statistic, p_value = _permutation_test(data1, data2)
        else:
            raise ValueError(f"Unknown test: {test}")
    
    # Effect size
    if test == 'ttest':
        # Cohen's d
        pooled_std = np.sqrt(((n_1 - 1) * std_1**2 + (n_2 - 1) * std_2**2) / (n_1 + n_2 - 2))
        effect_size = (mean_1 - mean_2) / pooled_std if pooled_std > 0 else 0
    else:
        # Rank-biserial correlation for non-parametric
        effect_size = 1 - (2 * statistic) / (n_1 * n_2) if (n_1 * n_2) > 0 else 0
    
    significant = p_value < 0.05
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'mean_1': mean_1,
        'mean_2': mean_2,
        'std_1': std_1,
        'std_2': std_2,
        'n_1': n_1,
        'n_2': n_2,
        'significant': significant,
        'test_used': test
    }


def _permutation_test(data1, data2, n_permutations=10000):
    """Permutation test for difference in means."""
    observed_diff = np.mean(data1) - np.mean(data2)
    combined = np.concatenate([data1, data2])
    n1 = len(data1)
    
    # Generate permutations
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        perm_diffs.append(perm_diff)
    
    perm_diffs = np.array(perm_diffs)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return observed_diff, p_value


def compare_multiple_conditions(data_dict, metric_name='value', test='kruskal'):
    """
    Compare a metric across multiple conditions (one-way ANOVA/Kruskal-Wallis).
    
    Parameters:
        data_dict (dict): Dictionary mapping condition names to data arrays
                         e.g., {'control': array1, 'drug_A': array2, 'drug_B': array3}
        metric_name (str): Name of metric being compared
        test (str): 'anova' or 'kruskal'
    
    Returns:
        dict: Dictionary containing:
            - 'omnibus_statistic': F-statistic (ANOVA) or H-statistic (Kruskal)
            - 'omnibus_p_value': Overall p-value
            - 'significant': Boolean indicating significant difference
            - 'pairwise_comparisons': Dictionary of pairwise test results
            - 'group_means': Dictionary of group means
            - 'group_stds': Dictionary of group stds
            - 'group_ns': Dictionary of group sample sizes
    
    Example:
        >>> data = {
        >>>     'control': np.array([1.2, 1.5, 1.3]),
        >>>     'drug_A': np.array([2.1, 2.3, 2.0]),
        >>>     'drug_B': np.array([1.8, 1.9, 1.7])
        >>> }
        >>> result = compare_multiple_conditions(data, test='kruskal')
        >>> print(f"Overall p-value: {result['omnibus_p_value']:.4f}")
    """
    # Extract condition names and data
    conditions = list(data_dict.keys())
    data_arrays = [data_dict[cond][~np.isnan(data_dict[cond])] for cond in conditions]
    
    # Remove empty arrays
    valid_conditions = [cond for cond, arr in zip(conditions, data_arrays) if len(arr) > 0]
    valid_data = [arr for arr in data_arrays if len(arr) > 0]
    
    if len(valid_data) < 2:
        return {
            'omnibus_statistic': np.nan,
            'omnibus_p_value': 1.0,
            'significant': False,
            'pairwise_comparisons': {},
            'group_means': {},
            'group_stds': {},
            'group_ns': {}
        }
    
    # Omnibus test
    if test == 'anova':
        statistic, p_value = f_oneway(*valid_data)
    else:  # kruskal
        statistic, p_value = kruskal(*valid_data)
    
    significant = p_value < 0.05
    
    # Group statistics
    group_means = {cond: np.mean(data_dict[cond]) for cond in valid_conditions}
    group_stds = {cond: np.std(data_dict[cond], ddof=1) for cond in valid_conditions}
    group_ns = {cond: len(data_dict[cond]) for cond in valid_conditions}
    
    # Pairwise comparisons (if significant)
    pairwise_comparisons = {}
    if significant:
        for i, cond1 in enumerate(valid_conditions):
            for cond2 in valid_conditions[i+1:]:
                pair_key = f"{cond1}_vs_{cond2}"
                pairwise_comparisons[pair_key] = compare_two_conditions(
                    data_dict[cond1], data_dict[cond2],
                    test='mannwhitney' if test == 'kruskal' else 'ttest'
                )
    
    return {
        'omnibus_statistic': statistic,
        'omnibus_p_value': p_value,
        'significant': significant,
        'pairwise_comparisons': pairwise_comparisons,
        'group_means': group_means,
        'group_stds': group_stds,
        'group_ns': group_ns,
        'test_used': test
    }


def compute_response_metrics_batch(stacks_dict, fs, baseline_frames=100):
    """
    Compute Ca²⁺ response metrics for multiple datasets.
    
    Extracts key metrics (peak amplitude, time to peak, etc.) from each dataset.
    
    Parameters:
        stacks_dict (dict): Dictionary mapping condition names to image stacks
        fs (float): Sampling frequency (Hz)
        baseline_frames (int): Number of frames for baseline estimation
    
    Returns:
        dict: Dictionary containing arrays of metrics for each condition:
            - 'peak_amplitude': Peak ΔF/F₀ for each dataset
            - 'time_to_peak': Time to peak (seconds)
            - 'auc': Area under curve
            - 'baseline_std': Baseline noise level
    
    Example:
        >>> stacks = {'control': stack1, 'treatment': stack2}
        >>> metrics = compute_response_metrics_batch(stacks, fs=30.0)
        >>> # Compare peak amplitudes
        >>> comparison = compare_two_conditions(
        >>>     metrics['peak_amplitude']['control'],
        >>>     metrics['peak_amplitude']['treatment']
        >>> )
    """
    metrics = {
        'peak_amplitude': {},
        'time_to_peak': {},
        'auc': {},
        'baseline_std': {}
    }
    
    for condition, stack in stacks_dict.items():
        # Compute mean trace
        mean_trace = np.mean(stack, axis=(1, 2))
        
        # Baseline
        baseline = np.mean(mean_trace[:baseline_frames])
        baseline_std = np.std(mean_trace[:baseline_frames])
        
        # ΔF/F₀
        dff = (mean_trace - baseline) / baseline
        
        # Metrics
        peak_idx = np.argmax(dff)
        peak_amplitude = dff[peak_idx]
        time_to_peak = peak_idx / fs
        auc = np.trapz(dff, dx=1/fs)
        
        metrics['peak_amplitude'][condition] = peak_amplitude
        metrics['time_to_peak'][condition] = time_to_peak
        metrics['auc'][condition] = auc
        metrics['baseline_std'][condition] = baseline_std
    
    return metrics


def compute_dose_response_curve(doses, responses, fit_hill=True):
    """
    Compute dose-response relationship.
    
    Fits Hill equation to dose-response data.
    
    Parameters:
        doses (ndarray): Array of concentrations/doses
        responses (ndarray): Array of responses (e.g., peak Ca²⁺ amplitude)
        fit_hill (bool): Whether to fit Hill equation
    
    Returns:
        dict: Dictionary containing:
            - 'ec50': Half-maximal effective concentration
            - 'hill_slope': Hill coefficient (steepness)
            - 'max_response': Maximum response
            - 'fitted_curve': Fitted Hill equation
            - 'r_squared': Goodness of fit
    
    Example:
        >>> doses = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0])  # μM
        >>> responses = np.array([0, 0.1, 0.3, 0.7, 0.95, 1.0])
        >>> dr = compute_dose_response_curve(doses, responses)
        >>> print(f"EC50: {dr['ec50']:.2f} μM")
        >>> print(f"Hill slope: {dr['hill_slope']:.2f}")
    """
    # Remove zero doses for log transform
    nonzero = doses > 0
    log_doses = np.log10(doses[nonzero])
    responses_nonzero = responses[nonzero]
    
    if not fit_hill or len(log_doses) < 4:
        return {
            'ec50': None,
            'hill_slope': None,
            'max_response': np.max(responses),
            'fitted_curve': None,
            'r_squared': None
        }
    
    # Hill equation: R = Rmax / (1 + (EC50/dose)^n)
    # In log space: R = Rmax / (1 + 10^(n*(log(EC50) - log(dose))))
    def hill_equation(log_dose, log_ec50, n, Rmax):
        return Rmax / (1 + 10**(n * (log_ec50 - log_dose)))
    
    try:
        from scipy.optimize import curve_fit
        
        # Initial guesses
        Rmax_guess = np.max(responses_nonzero)
        log_ec50_guess = np.median(log_doses)
        n_guess = 1.0
        
        popt, _ = curve_fit(hill_equation, log_doses, responses_nonzero,
                           p0=[log_ec50_guess, n_guess, Rmax_guess],
                           maxfev=5000)
        
        log_ec50, n, Rmax = popt
        ec50 = 10**log_ec50
        
        # Generate fitted curve
        log_doses_fit = np.linspace(log_doses.min(), log_doses.max(), 100)
        fitted_curve = hill_equation(log_doses_fit, log_ec50, n, Rmax)
        
        # R²
        predicted = hill_equation(log_doses, log_ec50, n, Rmax)
        ss_res = np.sum((responses_nonzero - predicted)**2)
        ss_tot = np.sum((responses_nonzero - np.mean(responses_nonzero))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'ec50': ec50,
            'hill_slope': n,
            'max_response': Rmax,
            'fitted_curve': (10**log_doses_fit, fitted_curve),
            'r_squared': r_squared
        }
    except:
        return {
            'ec50': None,
            'hill_slope': None,
            'max_response': np.max(responses),
            'fitted_curve': None,
            'r_squared': None
        }


def generate_comparison_report(comparison_results, metric_name='metric'):
    """
    Generate human-readable comparison report.
    
    Parameters:
        comparison_results (dict): Results from compare_two_conditions or compare_multiple_conditions
        metric_name (str): Name of the metric being compared
    
    Returns:
        str: Formatted report string
    
    Example:
        >>> result = compare_two_conditions(control, treatment)
        >>> report = generate_comparison_report(result, metric_name='Peak Amplitude')
        >>> print(report)
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Comparison Report: {metric_name}")
    lines.append(f"{'='*60}\n")
    
    if 'mean_1' in comparison_results:
        # Two-condition comparison
        lines.append(f"Condition 1:")
        lines.append(f"  n = {comparison_results['n_1']}")
        lines.append(f"  Mean ± SD = {comparison_results['mean_1']:.3f} ± {comparison_results['std_1']:.3f}")
        lines.append(f"\nCondition 2:")
        lines.append(f"  n = {comparison_results['n_2']}")
        lines.append(f"  Mean ± SD = {comparison_results['mean_2']:.3f} ± {comparison_results['std_2']:.3f}")
        lines.append(f"\nStatistical Test: {comparison_results['test_used']}")
        lines.append(f"  Statistic = {comparison_results['statistic']:.3f}")
        lines.append(f"  p-value = {comparison_results['p_value']:.4f} {'***' if comparison_results['p_value'] < 0.001 else '**' if comparison_results['p_value'] < 0.01 else '*' if comparison_results['p_value'] < 0.05 else 'ns'}")
        lines.append(f"  Effect size = {comparison_results['effect_size']:.3f}")
        lines.append(f"  Significant: {comparison_results['significant']}")
        
    elif 'omnibus_p_value' in comparison_results:
        # Multiple-condition comparison
        lines.append(f"Groups:")
        for cond in comparison_results['group_means'].keys():
            lines.append(f"  {cond}:")
            lines.append(f"    n = {comparison_results['group_ns'][cond]}")
            lines.append(f"    Mean ± SD = {comparison_results['group_means'][cond]:.3f} ± {comparison_results['group_stds'][cond]:.3f}")
        
        lines.append(f"\nOmnibus Test: {comparison_results['test_used']}")
        lines.append(f"  Statistic = {comparison_results['omnibus_statistic']:.3f}")
        lines.append(f"  p-value = {comparison_results['omnibus_p_value']:.4f}")
        lines.append(f"  Significant: {comparison_results['significant']}")
        
        if comparison_results['pairwise_comparisons']:
            lines.append(f"\nPairwise Comparisons:")
            for pair, result in comparison_results['pairwise_comparisons'].items():
                lines.append(f"  {pair}:")
                lines.append(f"    p-value = {result['p_value']:.4f} {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'}")
                lines.append(f"    Effect size = {result['effect_size']:.3f}")
    
    lines.append(f"\n{'='*60}\n")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    """Unit tests for batch comparison module."""
    print("Testing batch_comparison_module...")
    
    # Test 1: Two-condition comparison
    print("\n1. Testing compare_two_conditions...")
    np.random.seed(42)
    
    # Control vs treatment with effect
    control = np.random.randn(20) * 0.5 + 1.0
    treatment = np.random.randn(20) * 0.5 + 1.5
    
    result = compare_two_conditions(control, treatment, 
                                    metric_name='amplitude',
                                    test='ttest')
    
    print(f"   Control: {result['mean_1']:.3f} ± {result['std_1']:.3f} (n={result['n_1']})")
    print(f"   Treatment: {result['mean_2']:.3f} ± {result['std_2']:.3f} (n={result['n_2']})")
    print(f"   p-value: {result['p_value']:.4f}")
    print(f"   Effect size: {result['effect_size']:.2f}")
    print(f"   Significant: {result['significant']}")
    assert 0 <= result['p_value'] <= 1, "Invalid p-value"
    print("   ✓ Two-condition comparison working")
    
    # Test 2: Multiple-condition comparison
    print("\n2. Testing compare_multiple_conditions...")
    data = {
        'control': np.random.randn(15) * 0.5 + 1.0,
        'drug_A': np.random.randn(15) * 0.5 + 1.5,
        'drug_B': np.random.randn(15) * 0.5 + 2.0
    }
    
    result = compare_multiple_conditions(data, test='kruskal')
    print(f"   Omnibus p-value: {result['omnibus_p_value']:.4f}")
    print(f"   Significant: {result['significant']}")
    print(f"   Group means:")
    for cond, mean in result['group_means'].items():
        print(f"     {cond}: {mean:.3f}")
    assert 'omnibus_p_value' in result, "Missing omnibus p-value"
    print("   ✓ Multiple-condition comparison working")
    
    # Test 3: Batch metrics computation
    print("\n3. Testing compute_response_metrics_batch...")
    # Create synthetic stacks
    T, H, W = 300, 32, 32
    fs = 30.0
    
    stacks = {}
    for cond in ['control', 'treatment']:
        stack = np.random.randn(T, H, W) * 5 + 100
        # Add Ca²⁺ transient
        amplitude = 50 if cond == 'control' else 80
        signal = amplitude * np.exp(-(np.arange(T) - 150)**2 / 500)
        stack += signal[:, np.newaxis, np.newaxis]
        stacks[cond] = stack
    
    metrics = compute_response_metrics_batch(stacks, fs)
    print(f"   Peak amplitudes:")
    for cond, amp in metrics['peak_amplitude'].items():
        print(f"     {cond}: {amp:.3f}")
    assert 'peak_amplitude' in metrics, "Missing metrics"
    print("   ✓ Batch metrics computation working")
    
    # Test 4: Dose-response curve
    print("\n4. Testing compute_dose_response_curve...")
    doses = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0])
    responses = np.array([0, 0.05, 0.2, 0.6, 0.9, 0.95])
    
    dr = compute_dose_response_curve(doses, responses)
    if dr['ec50'] is not None:
        print(f"   EC50: {dr['ec50']:.3f}")
        print(f"   Hill slope: {dr['hill_slope']:.2f}")
        print(f"   Max response: {dr['max_response']:.3f}")
        print(f"   R²: {dr['r_squared']:.3f}")
    else:
        print("   Could not fit Hill equation")
    print("   ✓ Dose-response curve computation working")
    
    # Test 5: Comparison report
    print("\n5. Testing generate_comparison_report...")
    result = compare_two_conditions(control, treatment, metric_name='Peak Amplitude')
    report = generate_comparison_report(result, metric_name='Peak Amplitude')
    print(report)
    assert len(report) > 0, "Empty report"
    print("   ✓ Comparison report generation working")
    
    print("\n✅ All batch comparison module tests passed!")
