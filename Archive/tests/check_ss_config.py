from transformers import AutoConfig

model_id = "state-spaces/mamba2-130m"

def check_config():
    try:
        config = AutoConfig.from_pretrained(model_id)
        print(config)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_config()
