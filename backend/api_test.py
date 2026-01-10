"""
API Test Script - Test the MedivisionAI backend
"""

import requests
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_models_info():
    """Test the models info endpoint"""
    print("\n" + "="*60)
    print("Testing Models Info Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/models/info")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"\nModels Loaded: {data.get('models', [])}")
        print(f"Device: {data.get('device', 'unknown')}")
        print(f"\nModel Info:")
        for model, info in data.get('model_info', {}).items():
            print(f"  - {model}: {info}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analyze_endpoint(image_path=None):
    """Test the analyze endpoint"""
    print("\n" + "="*60)
    print("Testing Analyze Endpoint")
    print("="*60)
    
    if image_path is None:
        print("⚠️  No image provided. Skipping analysis test.")
        print("To test with an image, run:")
        print("  python api_test.py path/to/xray.jpg")
        return None
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        print(f"Uploading image: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'mode': 'unified'}
            
            response = requests.post(
                f"{API_BASE_URL}/api/analyze",
                files=files,
                data=data
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Analysis Complete!")
            print(f"\nPrediction: {result.get('prediction', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0):.2%}")
            
            if 'body_part_detection' in result:
                bp = result['body_part_detection']
                print(f"\nBody Part: {bp.get('body_part', 'N/A')}")
                print(f"  Confidence: {bp.get('confidence', 0):.2%}")
            
            if 'fracture_detection' in result:
                fd = result['fracture_detection']
                print(f"\nFracture: {fd.get('label', 'N/A')}")
                print(f"  Confidence: {fd.get('confidence', 0):.2%}")
            
            if 'gradcam_image' in result:
                print(f"\n✅ Grad-CAM heatmap generated")
                print(f"  (Base64 image: {len(result['gradcam_image'])} characters)")
            
            if 'all_probabilities' in result:
                print(f"\nAll Probabilities:")
                for label, prob in result['all_probabilities'].items():
                    print(f"  - {label}: {prob:.2%}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    import sys
    
    print("\n" + "="*60)
    print("MedivisionAI Backend API Test Suite")
    print("="*60)
    
    # Test 1: Health Check
    health_ok = test_health_check()
    
    # Test 2: Models Info
    models_ok = test_models_info()
    
    # Test 3: Analyze (if image provided)
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_ok = test_analyze_endpoint(image_path)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Models Info:  {'✅ PASS' if models_ok else '❌ FAIL'}")
    if analyze_ok is not None:
        print(f"Analyze API:  {'✅ PASS' if analyze_ok else '❌ FAIL'}")
    else:
        print(f"Analyze API:  ⚠️  SKIPPED (no image provided)")
    print("="*60)
    
    if health_ok and models_ok:
        print("\n✅ Backend is operational!")
        if analyze_ok is None:
            print("\n💡 Tip: Test with an image:")
            print("   python api_test.py path/to/xray.jpg")
    else:
        print("\n❌ Some tests failed. Check the backend server.")

if __name__ == "__main__":
    main()
