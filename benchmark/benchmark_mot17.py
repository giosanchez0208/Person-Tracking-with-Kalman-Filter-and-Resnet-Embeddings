"""
MOT17 Benchmark Suite for Person Tracking System
Evaluates tracking performance using standard MOT metrics.
"""

import os
import cv2
import numpy as np
import configparser
from collections import defaultdict
from pathlib import Path
import json
import time
from tqdm import tqdm

from python.tracking_pipeline import tracking_pipeline, tracker


class MOT17Parser:
    """Parse MOT17 dataset structure and ground truth."""
    
    def __init__(self, mot17_root):
        self.mot17_root = Path(mot17_root)
        
    def get_sequences(self, split='train'):
        """Get all sequence folders for a split (train/test)."""
        split_dir = self.mot17_root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Get unique sequences (ignore detector suffixes)
        sequences = []
        seen_base = set()
        for seq_dir in sorted(split_dir.iterdir()):
            if seq_dir.is_dir():
                # Extract base name (e.g., MOT17-02 from MOT17-02-FRCNN)
                base_name = '-'.join(seq_dir.name.split('-')[:2])
                if base_name not in seen_base:
                    seen_base.add(base_name)
                    # Use FRCNN version by default (most common)
                    frcnn_dir = split_dir / f"{base_name}-FRCNN"
                    if frcnn_dir.exists():
                        sequences.append(frcnn_dir)
        
        return sequences
    
    def parse_seqinfo(self, seq_path):
        """Parse seqinfo.ini file."""
        seqinfo_path = seq_path / 'seqinfo.ini'
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        
        info = {
            'name': config.get('Sequence', 'name'),
            'imDir': config.get('Sequence', 'imDir'),
            'frameRate': int(config.get('Sequence', 'frameRate')),
            'seqLength': int(config.get('Sequence', 'seqLength')),
            'imWidth': int(config.get('Sequence', 'imWidth')),
            'imHeight': int(config.get('Sequence', 'imHeight')),
            'imExt': config.get('Sequence', 'imExt')
        }
        return info
    
    def parse_gt(self, seq_path):
        """
        Parse ground truth annotations.
        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
        """
        gt_path = seq_path / 'gt' / 'gt.txt'
        gt_data = defaultdict(list)
        
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                cls = int(parts[7])
                visibility = float(parts[8])
                
                # Only include pedestrians (class 1) with visibility > 0
                if cls == 1 and visibility > 0:
                    gt_data[frame_id].append({
                        'id': track_id,
                        'bbox': [x, y, x + w, y + h],  # Convert to xyxy format
                        'conf': conf,
                        'visibility': visibility
                    })
        
        return gt_data
    
    def get_frame_path(self, seq_path, frame_num, info):
        """Get path to specific frame image."""
        img_dir = seq_path / info['imDir']
        # Frame numbers are 1-indexed and zero-padded to 6 digits
        frame_filename = f"{frame_num:06d}{info['imExt']}"
        return img_dir / frame_filename


class MOTMetrics:
    """Calculate standard MOT evaluation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.num_frames = 0
        self.num_matches = 0
        self.num_switches = 0
        self.num_false_positives = 0
        self.num_misses = 0
        self.num_fragmentations = 0
        self.total_distance = 0.0
        self.total_objects = 0
        
        # Track history for ID switches and fragmentations
        self.last_match = {}  # gt_id -> predicted_id
        self.track_last_seen = {}  # gt_id -> frame_num
    
    def compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes in xyxy format."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute_center_distance(self, bbox1, bbox2):
        """Compute center-to-center distance between two bboxes."""
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def match_frame(self, gt_objects, pred_objects, iou_threshold=0.5):
        """
        Match predicted objects to ground truth using Hungarian algorithm approximation.
        Returns matches, false positives, and misses.
        """
        from scipy.optimize import linear_sum_assignment
        
        if len(gt_objects) == 0 and len(pred_objects) == 0:
            return [], [], []
        
        if len(gt_objects) == 0:
            return [], list(range(len(pred_objects))), []
        
        if len(pred_objects) == 0:
            return [], [], list(range(len(gt_objects)))
        
        # Build IoU matrix
        iou_matrix = np.zeros((len(gt_objects), len(pred_objects)))
        for i, gt_obj in enumerate(gt_objects):
            for j, pred_obj in enumerate(pred_objects):
                iou_matrix[i, j] = self.compute_iou(gt_obj['bbox'], pred_obj['bbox'])
        
        # Hungarian assignment
        gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
        
        matches = []
        matched_gt = set()
        matched_pred = set()
        
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            if iou_matrix[gt_idx, pred_idx] >= iou_threshold:
                matches.append((gt_idx, pred_idx, iou_matrix[gt_idx, pred_idx]))
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
        
        false_positives = [i for i in range(len(pred_objects)) if i not in matched_pred]
        misses = [i for i in range(len(gt_objects)) if i not in matched_gt]
        
        return matches, false_positives, misses
    
    def update(self, frame_num, gt_objects, pred_objects, iou_threshold=0.5):
        """Update metrics for a single frame."""
        self.num_frames += 1
        self.total_objects += len(gt_objects)
        
        matches, false_positives, misses = self.match_frame(
            gt_objects, pred_objects, iou_threshold
        )
        
        self.num_matches += len(matches)
        self.num_false_positives += len(false_positives)
        self.num_misses += len(misses)
        
        # Calculate distances for matched objects
        for gt_idx, pred_idx, iou in matches:
            distance = self.compute_center_distance(
                gt_objects[gt_idx]['bbox'],
                pred_objects[pred_idx]['bbox']
            )
            self.total_distance += distance
            
            gt_id = gt_objects[gt_idx]['id']
            pred_id = pred_objects[pred_idx]['id']
            
            # Check for ID switches
            if gt_id in self.last_match:
                if self.last_match[gt_id] != pred_id:
                    self.num_switches += 1
            
            # Check for fragmentations (track reappeared after absence)
            if gt_id in self.track_last_seen:
                if frame_num - self.track_last_seen[gt_id] > 1:
                    self.num_fragmentations += 1
            
            self.last_match[gt_id] = pred_id
            self.track_last_seen[gt_id] = frame_num
    
    def compute_metrics(self):
        """Compute final metrics."""
        # MOTA (Multiple Object Tracking Accuracy)
        mota = 1.0 - (self.num_false_positives + self.num_misses + self.num_switches) / max(self.total_objects, 1)
        
        # MOTP (Multiple Object Tracking Precision) - average distance
        motp = self.total_distance / max(self.num_matches, 1)
        
        # Precision and Recall
        precision = self.num_matches / max(self.num_matches + self.num_false_positives, 1)
        recall = self.num_matches / max(self.total_objects, 1)
        
        # F1 Score
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
        
        metrics = {
            'MOTA': mota * 100,  # Convert to percentage
            'MOTP': motp,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1': f1 * 100,
            'ID_Switches': self.num_switches,
            'Fragmentations': self.num_fragmentations,
            'False_Positives': self.num_false_positives,
            'Misses': self.num_misses,
            'Matches': self.num_matches,
            'Total_Objects': self.total_objects,
            'Frames': self.num_frames
        }
        
        return metrics


def reset_tracker():
    """Reset global tracker state between sequences."""
    from python.tracking_pipeline import tracker
    tracker.curr_frame = None
    tracker.curr_bboxes = None
    tracker.curr_bbox_ids = None
    
    # Reset identify module's memory
    from python import identify
    identify.memory.curr_entities = None
    identify.next_id_counter = 0


def run_tracker_on_sequence(seq_path, parser, max_frames=None, visualize=False):
    """
    Run tracker on a MOT17 sequence and return predictions.
    """
    reset_tracker()
    
    info = parser.parse_seqinfo(seq_path)
    num_frames = info['seqLength']
    if max_frames:
        num_frames = min(num_frames, max_frames)
    
    predictions = defaultdict(list)
    processing_times = []
    
    print(f"\n  Processing {info['name']} ({num_frames} frames)...")
    
    for frame_num in tqdm(range(1, num_frames + 1), desc="  Frames"):
        frame_path = parser.get_frame_path(seq_path, frame_num, info)
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            print(f"    Warning: Could not read frame {frame_num}")
            continue
        
        # Run tracking
        start_time = time.time()
        result_frame = tracking_pipeline(frame)
        elapsed = time.time() - start_time
        processing_times.append(elapsed)
        
        # Extract predictions from tracker state
        from python.tracking_pipeline import tracker as global_tracker
        if global_tracker.curr_bboxes:
            for i, bbox in enumerate(global_tracker.curr_bboxes):
                bbox_id = global_tracker.curr_bbox_ids.get(i, i)
                x1, y1, x2, y2, conf = bbox
                predictions[frame_num].append({
                    'id': int(bbox_id),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'conf': float(conf)
                })
        
        # Optional visualization
        if visualize and result_frame is not None:
            cv2.imshow('Tracking', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if visualize:
        cv2.destroyAllWindows()
    
    avg_fps = 1.0 / np.mean(processing_times) if processing_times else 0
    print(f"  Average FPS: {avg_fps:.2f}")
    
    return predictions, processing_times


def evaluate_sequence(seq_path, parser, max_frames=None, visualize=False):
    """Evaluate tracker on a single sequence."""
    print(f"\nEvaluating sequence: {seq_path.name}")
    
    # Parse ground truth
    gt_data = parser.parse_gt(seq_path)
    
    # Run tracker
    predictions, processing_times = run_tracker_on_sequence(
        seq_path, parser, max_frames, visualize
    )
    
    # Compute metrics
    metrics_calculator = MOTMetrics()
    
    for frame_num in sorted(gt_data.keys()):
        if max_frames and frame_num > max_frames:
            break
        
        gt_objects = gt_data.get(frame_num, [])
        pred_objects = predictions.get(frame_num, [])
        
        metrics_calculator.update(frame_num, gt_objects, pred_objects)
    
    metrics = metrics_calculator.compute_metrics()
    metrics['Avg_FPS'] = 1.0 / np.mean(processing_times) if processing_times else 0
    metrics['Avg_Frame_Time_ms'] = np.mean(processing_times) * 1000 if processing_times else 0
    
    return metrics, predictions


def benchmark_mot17(mot17_root, split='train', max_frames=None, visualize=False, output_dir='benchmark_results'):
    """
    Run full benchmark on MOT17 dataset.
    """
    parser = MOT17Parser(mot17_root)
    sequences = parser.get_sequences(split)
    
    print(f"\n{'='*60}")
    print(f"MOT17 Benchmark - {split.upper()} split")
    print(f"Found {len(sequences)} sequences")
    print(f"{'='*60}")
    
    all_results = {}
    
    for seq_path in sequences:
        try:
            metrics, predictions = evaluate_sequence(
                seq_path, parser, max_frames, visualize
            )
            all_results[seq_path.name] = metrics
            
            # Print sequence results
            print(f"\n  Results for {seq_path.name}:")
            print(f"    MOTA: {metrics['MOTA']:.2f}%")
            print(f"    MOTP: {metrics['MOTP']:.2f}")
            print(f"    Precision: {metrics['Precision']:.2f}%")
            print(f"    Recall: {metrics['Recall']:.2f}%")
            print(f"    F1: {metrics['F1']:.2f}%")
            print(f"    ID Switches: {metrics['ID_Switches']}")
            print(f"    Fragmentations: {metrics['Fragmentations']}")
            print(f"    Avg FPS: {metrics['Avg_FPS']:.2f}")
            
        except Exception as e:
            print(f"\n  Error processing {seq_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute overall metrics
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    
    if all_results:
        overall = compute_overall_metrics(all_results)
        
        print(f"\nAverage Metrics Across All Sequences:")
        print(f"  MOTA: {overall['MOTA']:.2f}%")
        print(f"  MOTP: {overall['MOTP']:.2f}")
        print(f"  Precision: {overall['Precision']:.2f}%")
        print(f"  Recall: {overall['Recall']:.2f}%")
        print(f"  F1: {overall['F1']:.2f}%")
        print(f"  Total ID Switches: {overall['ID_Switches']}")
        print(f"  Total Fragmentations: {overall['Fragmentations']}")
        print(f"  Average FPS: {overall['Avg_FPS']:.2f}")
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results_file = output_path / f'mot17_{split}_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'sequences': all_results,
                'overall': overall
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return all_results, overall
    
    return {}, {}


def compute_overall_metrics(all_results):
    """Compute weighted average metrics across sequences."""
    total_frames = sum(r['Frames'] for r in all_results.values())
    
    # Weighted averages
    overall = {
        'MOTA': sum(r['MOTA'] * r['Frames'] for r in all_results.values()) / total_frames,
        'MOTP': sum(r['MOTP'] * r['Frames'] for r in all_results.values()) / total_frames,
        'Precision': sum(r['Precision'] * r['Frames'] for r in all_results.values()) / total_frames,
        'Recall': sum(r['Recall'] * r['Frames'] for r in all_results.values()) / total_frames,
        'F1': sum(r['F1'] * r['Frames'] for r in all_results.values()) / total_frames,
        'ID_Switches': sum(r['ID_Switches'] for r in all_results.values()),
        'Fragmentations': sum(r['Fragmentations'] for r in all_results.values()),
        'False_Positives': sum(r['False_Positives'] for r in all_results.values()),
        'Misses': sum(r['Misses'] for r in all_results.values()),
        'Avg_FPS': np.mean([r['Avg_FPS'] for r in all_results.values()]),
        'Total_Frames': total_frames
    }
    
    return overall


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark tracker on MOT17 dataset')
    parser.add_argument('--mot17-root', type=str, default='MOT17',
                        help='Path to MOT17 dataset root')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max frames per sequence (for quick testing)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show tracking visualization')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    results, overall = benchmark_mot17(
        args.mot17_root,
        split=args.split,
        max_frames=args.max_frames,
        visualize=args.visualize,
        output_dir=args.output_dir
    )
