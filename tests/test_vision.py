import os
import yaml
from dotenv import load_dotenv
import litellm

def main():
    # Load environment variables
    load_dotenv()

    config_path = os.path.join("src", "llm", "config.dev.yaml")
    
    # Read config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    models = set()
    
    # Models in providers
    providers = config.get("router", {}).get("providers", {})
    for provider, data in providers.items():
        for model_entry in data.get("models", []):
            if "model" in model_entry:
                models.add(model_entry["model"])

    # Models in fallback_providers
    fallback_providers = config.get("router", {}).get("fallback_providers", {})
    for provider, data in fallback_providers.items():
        for model_entry in data.get("models", []):
            if "model" in model_entry:
                models.add(model_entry["model"])

    print(f"Checking vision support for {len(models)} unique models...")
    print("-" * 50)
    
    for model in sorted(models):
        try:
            # litellm.supports_vision() returns True/False
            supports = litellm.supports_vision(model=model)
            print(f"[{'YES' if supports else 'NO '}] {model}")
        except Exception as e:
            print(f"[ERR] {model} - {str(e)}")

if __name__ == "__main__":
    main()
