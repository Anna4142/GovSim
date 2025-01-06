import wandb
import uuid

def test_wandb_logging():
    try:
        # Initialize a wandb run
        run = wandb.init(
            project="test_project",
            name=f"test_run_{uuid.uuid4()}",
            config={
                "test_key": "test_value"
            }
        )
        
        # Log some metrics
        for i in range(5):
            wandb.log({
                "iteration": i,
                "loss": 1.0 / (i + 1),
                "accuracy": 0.9 ** i
            })
        
        # Finish the run
        wandb.finish()
        
        print("Wandb logging test completed successfully!")
    
    except Exception as e:
        print(f"Wandb logging failed with error: {e}")

if __name__ == "__main__":
    test_wandb_logging()