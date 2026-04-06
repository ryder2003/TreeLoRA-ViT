import torch
from datasets import get_split_cifar100
from continual_learner import TreeLoRALearner
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda')
task_dataloaders, _ = get_split_cifar100('./data', n_tasks=2, batch_size=16, num_workers=0)
learner = TreeLoRALearner(num_tasks=2, classes_per_task=50, device=device, pretrained=False)

# Override tqdm in train_task to avoid spam
def tqdm_override(iterable, *args, **kwargs):
    return iterable
import continual_learner
continual_learner.tqdm = tqdm_override

print("Starting training...")
try:
    learner.train_task(task_id=0, train_loader=task_dataloaders[0][0], epochs=1)
    print("\nSUCCESS!")
except Exception as e:
    import traceback
    traceback.print_exc()
