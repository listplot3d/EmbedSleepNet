import argparse
import time
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from model import EmbedSleepNet

def generate_compatible_data(batch_size, seq_length):
    """Generate data compatible with the model's input layer"""
    # Adjust dimensions according to the model structure requirements
    # The original model might expect an input shape of (batch, 1, 3000, 1) similar to a 4D tensor
    data = torch.randn(batch_size, 1, seq_length, 1)  # Add the last dimension
    return data.permute(0, 1, 3, 2)  # Rearrange to (batch, channel, height, width)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-rate', type=float, default=1.0,
                      help='Data generation rate (samples/second)')
    parser.add_argument('--seq-length', type=int, default=3000,
                      help='Input sequence length (must match training parameters)')
    args = parser.parse_args()

    # Model loading
    model_path = Path(__file__).parent/'test_EmbedSleepNet_model.pth'
    model = EmbedSleepNet()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    except Exception as e:
        print(f"Failed to load the model: {str(e)}")
        return

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"\n{'='*40}")
    print(f"Device type: {device}")
    print(f"Adjusted input specifications: batch_size=1, channels=1, height=1, width={args.seq_length}")
    print(f"{'='*40}\n")

    try:
        with torch.no_grad():
            while True:
                cycle_start = time.perf_counter()
                
                # Generate compatible data
                fake_data = generate_compatible_data(1, args.seq_length).to(device)
                
                # Dimension validation
                # print(f"Input data shape: {fake_data.shape}")
                assert fake_data.shape == (1, 1, 1, args.seq_length), \
                    f"Dimension mismatch! Expected (1,1,1,{args.seq_length}) but got {fake_data.shape}"
                
                # Inference
                start_infer = time.perf_counter()
                output = model(fake_data)
                print(fake_data.shape, fake_data.dtype)
                infer_time = time.perf_counter() - start_infer
                
                # Display basic results
                print(f"[{time.strftime('%H:%M:%S')}] "
                     f"Inference successful! Time taken: {infer_time*1000:.1f}ms"
                     f"{output}"
                     )
                
                # Rate control
                elapsed = time.perf_counter() - cycle_start
                time.sleep(max(0.0, (1.0 / args.data_rate) - elapsed))
                
    except KeyboardInterrupt:
        print("\nTest terminated successfully")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Suggested solutions:")
        print("1. Check the input shape expected by the model")
        print("2. Verify the data format used during training")
        print("3. Contact the model developer for input specifications")

if __name__ == "__main__":
    main()
