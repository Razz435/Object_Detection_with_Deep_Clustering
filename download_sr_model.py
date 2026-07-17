import os
import urllib.request

def download_sr_model():
    """Download EDSR super-resolution model (optional)"""
    model_dir = 'sr_models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"
    model_path = os.path.join(model_dir, "EDSR_x2.pb")
    
    if not os.path.exists(model_path):
        print("Downloading EDSR super-resolution model...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
    else:
        print("Super-resolution model already exists.")
    
    return model_path

if __name__ == "__main__":
    download_sr_model()