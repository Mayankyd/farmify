from huggingface_hub import hf_hub_download
import pickle
import os

def load_model_from_hf():
    # Download the model from Hugging Face
    model_path = hf_hub_download(
        repo_id="AxilBlaze2036/farmify",
        filename="model.h5"
    )
    
    # Create a models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Copy the model to the models directory
    import shutil
    shutil.copy(model_path, 'models/model.h5')
    print(f"Model downloaded and saved to: models/model.h5")
    return model_path

if __name__ == "__main__":
    try:
        model_path = load_model_from_hf()
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {str(e)}") 