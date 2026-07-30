import torch
import matplotlib.pyplot as plt
import os

def check_singular_spectrum():
    # 1. Load the pre-trained HDC classification weights
    ckpt_path = 'logs/kitti_pretrain/hdc_sub.pth'
    if not os.path.exists(ckpt_path):
        print(f"Cannot find {ckpt_path}. Please run this where the weights are located.")
        return

    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 2. Extract the 10,000D prototypes
    # Assuming the weights are stored under 'classify.weight'
    if 'classify.weight' in ckpt:
        W = ckpt['classify.weight']
    elif 'state_dict' in ckpt and 'classify.weight' in ckpt['state_dict']:
        W = ckpt['state_dict']['classify.weight']
    else:
        print("Could not find 'classify.weight' in the checkpoint.")
        return

    print(f"Prototype Matrix Shape: {W.shape}") # Should be (17, 10000)
    
    # 3. Compute Singular Value Decomposition
    # W is (num_classes, hd_dim)
    # U will be (17, 17), S will be (17,), V will be (10000, 10000) but we only need S
    U, S, V = torch.svd(W.float())
    
    print("\nSingular Values (Spectrum):")
    for i, s in enumerate(S):
        print(f"  sigma_{i+1}: {s.item():.4f}")
        
    # Calculate energy retention
    total_energy = torch.sum(S ** 2)
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
    
    print("\nCumulative Energy Retention:")
    for i, e in enumerate(cumulative_energy):
        print(f"  Top {i+1} components explain: {e.item()*100:.2f}% of variance")
        
    # 4. Save a plot of the spectrum for easy visualization
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(S) + 1), S.numpy(), marker='o', linestyle='-', color='b')
    plt.title('Singular Value Spectrum of 10K-D Prototypes')
    plt.xlabel('Component Index')
    plt.ylabel('Singular Value Magnitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_img = 'logs/n4_svd_spectrum.png'
    os.makedirs('logs', exist_ok=True)
    plt.savefig(out_img)
    print(f"\nSaved spectrum plot to {out_img}")
    
    # 5. Conclusion Logic
    if cumulative_energy[0] < 0.2:
        print("\n[DIAGNOSTIC] The spectrum is extremely flat. HDC is holographic; D-optimal compression (N4) will likely destroy the signal.")
    elif cumulative_energy[4] > 0.8:
        print("\n[DIAGNOSTIC] The spectrum decays rapidly. A dense core exists; D-optimal compression (N4) is highly viable.")
    else:
        print("\n[DIAGNOSTIC] The spectrum decays moderately. N4 might work, but check the plot.")

if __name__ == '__main__':
    check_singular_spectrum()
