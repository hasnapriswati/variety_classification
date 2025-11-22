import os
import json
import time
import requests
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from collections import defaultdict
import concurrent.futures
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, api_url: str = "http://localhost:5000/predict"):
        self.api_url = api_url
        self.class_names = [
            "Branang", "Carla_agrihorti", "Carvi_agrihorti", "Ciko", "Hot_beauty",
            "Hot_vision", "Inata_agrihorti", "Ivegri", "Leaf_Tanjung", "Lingga",
            "Mia", "Pertiwi", "Pilar"
        ]
        self.results = []
        
    def extract_label_from_filename(self, csv_line: str) -> Tuple[str, np.ndarray]:
        """Extract filename and label vector from CSV line"""
        parts = csv_line.strip().split(',')
        filename = parts[0]
        label_vector = np.array([int(x) for x in parts[1:]])
        return filename, label_vector
    
    def get_true_label(self, label_vector: np.ndarray) -> int:
        """Get the index of the true class (where value is 1)"""
        return np.argmax(label_vector)
    
    def predict_image(self, image_path: str, max_retries: int = 3) -> Dict:
        """Send image to prediction API with retry logic"""
        for attempt in range(max_retries):
            try:
                with open(image_path, 'rb') as f:
                    files = {'image': f}
                    response = requests.post(self.api_url, files=files, timeout=30)
                    
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Attempt {attempt + 1}: API returned status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}: Request failed - {e}")
                
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                
        logger.error(f"Failed to get prediction for {image_path} after {max_retries} attempts")
        return None
    
    def process_dataset(self, dataset_dir: str, csv_file: str) -> Dict[str, float]:
        """Process a dataset and calculate metrics"""
        logger.info(f"Processing dataset: {dataset_dir}")
        
        # Read CSV file
        csv_path = os.path.join(dataset_dir, csv_file)
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return {}
        
        y_true = []
        y_pred = []
        confidences = []
        processing_times = []
        
        # Read CSV lines
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header
        
        logger.info(f"Found {len(lines)} images to process")
        
        # Process each image
        for i, line in enumerate(lines):
            if i % 50 == 0:
                logger.info(f"Processing image {i+1}/{len(lines)}")
                
            try:
                filename, label_vector = self.extract_label_from_filename(line)
                true_label = self.get_true_label(label_vector)
                
                image_path = os.path.join(dataset_dir, filename)
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Get prediction
                start_time = time.time()
                prediction = self.predict_image(image_path)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                if prediction and prediction.get('success'):
                    predicted_variety = prediction.get('variety')
                    confidence = float(prediction.get('confidence_percentage', '0').rstrip('%'))
                    
                    # Map variety name to index
                    if predicted_variety in self.class_names:
                        predicted_label = self.class_names.index(predicted_variety)
                    else:
                        logger.warning(f"Unknown variety: {predicted_variety}")
                        continue
                    
                    y_true.append(true_label)
                    y_pred.append(predicted_label)
                    confidences.append(confidence)
                    
                    # Store detailed result
                    self.results.append({
                        'filename': filename,
                        'true_label': int(true_label),
                        'true_variety': self.class_names[true_label],
                        'predicted_label': int(predicted_label),
                        'predicted_variety': predicted_variety,
                        'confidence': confidence,
                        'processing_time': processing_time,
                        'correct': bool(true_label == predicted_label)
                    })
                    
                else:
                    logger.warning(f"Prediction failed for {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        # Calculate metrics
        if len(y_true) == 0:
            logger.error("No successful predictions to calculate metrics")
            return {}
        
        # Calculate overall metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        
        # Calculate per-class metrics
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            output_dict=True, 
            zero_division=0
        )
        
        # Calculate average confidence and processing time
        avg_confidence = np.mean(confidences) if confidences else 0
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time,
            'total_images': len(lines),
            'successful_predictions': len(y_true),
            'class_report': class_report
        }
        
        logger.info(f"Successfully processed {len(y_true)}/{len(lines)} images")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        return metrics
    
    def convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        return obj
    
    def save_results(self, output_file: str = 'evaluation_results.json'):
        """Save evaluation results to JSON file"""
        results_data = {
            'class_names': self.class_names,
            'results': self.results,
            'summary': {
                'total_predictions': len(self.results),
                'correct_predictions': sum(1 for r in self.results if r['correct']),
                'overall_accuracy': float(np.mean([r['correct'] for r in self.results])) if self.results else 0
            }
        }
        
        # Convert to JSON serializable format
        results_data = self.convert_to_json_serializable(results_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
    
    def print_detailed_report(self, metrics: Dict):
        """Print detailed evaluation report"""
        print("\n" + "="*80)
        print("EVALUATION REPORT - INTEGRATED RICE VARIETY CLASSIFICATION MODEL")
        print("="*80)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision:          {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"   Recall:             {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"   F1-Score:           {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"   Avg Confidence:     {metrics['avg_confidence']:.2f}%")
        print(f"   Avg Processing Time: {metrics['avg_processing_time']:.3f}s")
        
        print(f"\n📈 DATASET STATISTICS:")
        print(f"   Total Images:       {metrics['total_images']}")
        print(f"   Successful Predictions: {metrics['successful_predictions']}")
        print(f"   Success Rate:       {(metrics['successful_predictions']/metrics['total_images']*100):.2f}%")
        
        if 'class_report' in metrics:
            print(f"\n🏷️  PER-CLASS METRICS:")
            print("-"*80)
            print(f"{'Class Name':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-"*80)
            
            for class_name in self.class_names:
                if class_name in metrics['class_report']:
                    report = metrics['class_report'][class_name]
                    print(f"{class_name:<20} {report['precision']:<10.4f} {report['recall']:<10.4f} "
                          f"{report['f1-score']:<10.4f} {int(report['support']):<10}")
            
            print("-"*80)
            
        print("\n" + "="*80)

def main():
    """Main evaluation function"""
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Define dataset paths
    dataset_base = "D:\\DOCUMENT\\Hasna punya\\Semester 7\\variety_classification\\data\\Dataset project.v6i.multiclass"
    split = os.environ.get("EVAL_SPLIT", "both").strip().lower()
    valid_dir = os.path.join(dataset_base, "valid")
    test_dir = os.path.join(dataset_base, "test")
    print("🚀 Starting model evaluation...")
    print(f"API URL: {evaluator.api_url}")
    
    results = {}
    if split in ("both", "valid") and os.path.exists(valid_dir):
        print(f"Dataset: {valid_dir}")
        metrics = evaluator.process_dataset(valid_dir, "_classes.csv")
        if metrics:
            evaluator.print_detailed_report(metrics)
            evaluator.save_results("validation_evaluation_results.json")
            results.update({
                'validation_accuracy': metrics['accuracy'],
                'validation_precision': metrics['precision'],
                'validation_recall': metrics['recall'],
                'validation_f1': metrics['f1_score']
            })
    if split in ("both", "test") and os.path.exists(test_dir):
        print(f"\n🔄 Processing test dataset...")
        test_metrics = evaluator.process_dataset(test_dir, "_classes.csv")
        if test_metrics:
            print(f"\n📊 TEST DATASET RESULTS:")
            evaluator.print_detailed_report(test_metrics)
            evaluator.save_results("test_evaluation_results.json")
            results.update({
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1_score']
            })
    if results:
        print(f"\n✅ Evaluation completed successfully!")
        return results
    print(f"\n❌ Evaluation failed - no metrics calculated")
    return None

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n🎯 KEY RESULTS:")
        print(f"   Validation Accuracy: {results['validation_accuracy']:.4f}")
        print(f"   Validation Precision: {results['validation_precision']:.4f}")
        print(f"   Validation Recall: {results['validation_recall']:.4f}")
        print(f"   Validation F1-Score: {results['validation_f1']:.4f}")
