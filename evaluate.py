import torch
from pathlib import Path
from model import create_model
from dataset import TahoeDataset, get_dataloader
from trainer import Trainer
import torch.utils.tensorboard as tensorboard
from datetime import datetime

def find_latest_checkpoint(checkpoint_dir):
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    return latest_checkpoint

def main():
   
    checkpoint_dir = "checkpoints"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dataset
    dataset = TahoeDataset(
        data_source="hf",
        split="train",  
        gene_max_cnt=1000
    )
    

    model = create_model(
        input_dim=1000,
        d_model=64,
        nhead=4,
        num_layers=4,
        num_classes=3
    )
    

    trainer = Trainer(
        model=model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        run_name=datetime.now().strftime("%Y%m%d_%H%M%S") + "_eval"
    )
    
   
    train_loader, val_loader, test_loader = trainer.create_data_loader(
        dataset,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader
    trainer.test_loader = test_loader
    
    try:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        print(f"\nFound latest checkpoint: {latest_checkpoint}")
        
        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from step {checkpoint['global_step']}")
        
        # Set up TensorBoard writer
        writer = tensorboard.SummaryWriter(log_dir=trainer.run_dir / 'tensorboard')
        
        # Run evaluation on test set
        print("\nRunning evaluation on test set...")
        model.eval()
        test_loss, test_acc = trainer._evaluate(test_loader, prefix="test", num_batches=None)
        
        # Log metrics
        print(f"\nTest Results:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar("test/final_loss", test_loss, checkpoint['global_step'])
        writer.add_scalar("test/final_accuracy", test_acc, checkpoint['global_step'])
        
        # Close TensorBoard writer
        writer.close()
        
        print(f"\nResults have been logged to TensorBoard in {trainer.run_dir / 'tensorboard'}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main() 