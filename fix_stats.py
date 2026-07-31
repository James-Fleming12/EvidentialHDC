import re

with open('unsup_kitti-c.py', 'r') as f:
    content = f.read()

# Extract the function body
start_idx = content.find("def populate_source_statistics(model, data_dir, arch_cfg, data_cfg, device, dry_run=False):")
end_idx = content.find("def main():", start_idx)

func_body = content[start_idx:end_idx]

# Replace `model.` with `self.`
method_body = func_body.replace("model.", "self.")
method_body = method_body.replace("def populate_source_statistics(self, data_dir, arch_cfg, data_cfg, device, dry_run=False):", "    def populate_source_statistics(self, data_dir, arch_cfg, data_cfg, device, dry_run=False):\n        from dataset.kitti.parser import Parser\n        from torch.utils.data import DataLoader")

# Add indent
method_body = '\n'.join(['    ' + line if line.strip() and not line.startswith('    def populate_') else line for line in method_body.split('\n')])
# Fix the def line specifically
method_body = method_body.replace("    def populate_source_statistics(self, data_dir, arch_cfg, data_cfg, device, dry_run=False):", "    def populate_source_statistics(self, data_dir, arch_cfg, data_cfg, device, dry_run=False):")

# Now inject it into DualGateModel
with open('modules/HDC_utils.py', 'r') as f:
    hdc_content = f.read()
    
inject_idx = hdc_content.find("    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None):")
new_hdc_content = hdc_content[:inject_idx] + method_body + "\n" + hdc_content[inject_idx:]

with open('modules/HDC_utils.py', 'w') as f:
    f.write(new_hdc_content)

# Now delete it from unsup_kitti-c.py
new_content = content[:start_idx] + content[end_idx:]
with open('unsup_kitti-c.py', 'w') as f:
    f.write(new_content)

# Now update the call site in unsup_kitti-c.py
with open('unsup_kitti-c.py', 'r') as f:
    content = f.read()

content = content.replace("populate_source_statistics(base_model, args.kitti_dir, ARCH, DATA, device, dry_run=args.dry_run)", "base_model.populate_source_statistics(args.kitti_dir, ARCH, DATA, device, dry_run=args.dry_run)")

with open('unsup_kitti-c.py', 'w') as f:
    f.write(content)
