import requests
import json
import os

def test_predict(image_path):
    """Test the predict endpoint with an image"""
    url = "http://localhost:5000/predict"
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                result = response.json()
                print("✅ Prediction successful!")
                print(f"Variety: {result.get('variety', 'Unknown')}")
                print(f"Confidence: {result.get('confidence_percentage', '0%')}")
                print(f"Decision Rule: {result.get('decision_rule', 'unknown')}")
                
                # Print morphology info if available
                morph_info = result.get('morphology_info', {})
                if morph_info:
                    print("\n📊 Morphology Info:")
                    for key, value in morph_info.items():
                        print(f"  {key}: {value}")
                
                # Print YOLO detections
                yolo_info = result.get('yolo', {})
                detections = yolo_info.get('detections', [])
                if detections:
                    print(f"\n🔍 YOLO Detections: {len(detections)} leaf(s) found")
                    for i, det in enumerate(detections):
                        print(f"  Detection {i+1}: {det.get('class_name', 'unknown')} (confidence: {det.get('confidence', 0):.2f})")
                
                return result
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None

if __name__ == "__main__":
    # Test with sample images from the dataset
    test_images = [
        "D:\\DOCUMENT\\Hasna punya\\Semester 7\\variety_classification\\data\\Dataset project.v6i.multiclass\\valid\\0720A1_jpg.rf.9d223f84e1da681698fc9ee653f1ce65.jpg",
        "D:\\DOCUMENT\\Hasna punya\\Semester 7\\variety_classification\\data\\Dataset project.v6i.multiclass\\valid\\20220726_083444b_jpg.rf.ab4d87b041aba2688c3b11f7618c6e0a.jpg",
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            print(f"Testing with: {os.path.basename(image_path)}")
            print(f"{'='*60}")
            test_predict(image_path)
            # Test both images instead of breaking after first
            continue
        else:
            print(f"Image not found: {image_path}")
    
    print(f"\n{'='*60}")
    print("Testing completed!")