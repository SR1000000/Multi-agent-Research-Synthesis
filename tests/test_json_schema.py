import os
import yaml
from dotenv import load_dotenv
import litellm

# Not definitive.  Most models support json_schema but aren't listed as such in litellm's static registry.

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

    print(f"Checking response_format and json_schema support for {len(models)} unique models...")
    print("-" * 80)
    print(f"{'MODEL':<50} | {'response_format'} | {'json_schema'}")
    print("-" * 80)
    
    for model in sorted(models):
        try:
            params = litellm.get_supported_openai_params(model=model)
            supports_response_format = "response_format" in params if params else False
            
            supports_schema = litellm.supports_response_schema(model=model)
            
            rf_str = "YES" if supports_response_format else "NO "
            js_str = "YES" if supports_schema else "NO "
            
            print(f"{model:<50} | {rf_str:<15} | {js_str}")
        except Exception as e:
            print(f"[ERR] {model:<44} | {str(e)}")

if __name__ == "__main__":
    main()
