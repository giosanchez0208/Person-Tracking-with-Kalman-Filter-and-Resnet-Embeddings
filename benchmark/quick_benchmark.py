"""
Quick benchmark script - runs a fast test on a few frames per sequence.
"""

from benchmark.benchmark_mot17 import benchmark_mot17
from benchmark.visualize_results import generate_all_visualizations

if __name__ == '__main__':
    print("\n" + "="*80)
    print(" MOT17 QUICK BENCHMARK - Testing on first 100 frames per sequence")
    print("="*80)
    
    # Run quick benchmark (100 frames per sequence)
    results, overall = benchmark_mot17(
        mot17_root='MOT17',
        split='train',
        max_frames=100,
        visualize=False,
        output_dir='benchmark_results'
    )
    
    # Generate visualizations
    if results:
        print("\n" + "="*80)
        print(" GENERATING VISUALIZATIONS")
        print("="*80)
        
        generate_all_visualizations(
            'benchmark_results/mot17_train_results.json',
            'benchmark_results'
        )
        
        print("\n" + "="*80)
        print(" BENCHMARK COMPLETE!")
        print(" Check 'benchmark_results/' folder for detailed plots and tables")
        print("="*80 + "\n")
