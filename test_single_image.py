import requests
import json

def test_single_image():
    """Test single image with detailed voting analysis"""
    
    url = "http://localhost:5000/predict"
    image_path = "backend/uploads/0703B1.jpg"
    
    print("=== TEST SINGLE IMAGE: 0703B1.jpg ===\n")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"🎯 HASIL AKHIR:")
            print(f"   Varietas: {result.get('variety')}")
            print(f"   Confidence: {result.get('confidence_percentage')}")
            print(f"   Metode: {result.get('prediction_sources', {}).get('decision_method', 'N/A')}")
            
            if result.get('prediction_sources'):
                sources = result['prediction_sources']
                print(f"\n📊 DETAIL VOTING:")
                print(f"   XGBoost: {sources.get('xgboost_variety')} ({sources.get('xgboost_confidence', 0)*100:.1f}%)")
                print(f"   Morfologi: {sources.get('morphology_variety')} ({sources.get('morphology_confidence', 0)*100:.1f}%)")
                
                # Analisis detail
                xgboost_variety = sources.get('xgboost_variety')
                morphology_variety = sources.get('morphology_variety')
                final_variety = result.get('variety')
                decision_method = sources.get('decision_method')
                
                print(f"\n🔍 ANALISIS:")
                if decision_method == 'consensus':
                    print(f"   ✅ Konsensus tercapai antara XGBoost dan morfologi")
                else:
                    print(f"   🔄 Voting dilakukan karena perbedaan prediksi")
                    if final_variety != xgboost_variety:
                        print(f"   📊 Morfologi mengalahkan XGBoost!")
                        if xgboost_variety == 'Ciko':
                            print(f"   🎯 Bias Ciko berhasil diatasi!")
                    else:
                        print(f"   📈 XGBoost dipilih berdasarkan confidence")
                
                print(f"\n📏 DETAIL MORFOLOGI:")
                morph_info = result.get('morphology_info', {})
                print(f"   Panjang: {morph_info.get('panjang_daun_mm', 0):.1f} mm")
                print(f"   Lebar: {morph_info.get('lebar_daun_mm', 0):.1f} mm")
                print(f"   Rasio Bentuk: {morph_info.get('rasio_bentuk_daun', 0):.2f}")
                print(f"   Interpretasi: {morph_info.get('interpretasi_rasio_bentuk', {}).get('bentuk', 'N/A')}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    test_single_image()