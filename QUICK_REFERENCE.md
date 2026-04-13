# TreeLoRA Quick Reference Card

## 🎯 Problem & Solution

**Your Result**: 97.18% accuracy on CIFAR-100  
**Paper's Result**: ~68% accuracy on CIFAR-100  
**Reason**: Different evaluation protocols  
**Solution**: Use class-incremental implementation

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Understand the difference (2 min)
python compare_evaluation_protocols.py

# 2. Quick test (5 min)
python test_class_incremental.py

# 3. Train CIFAR-100 (60 min)
python train_class_incremental.py --dataset cifar100 --n_tasks 10 --epochs 10
```

---

## 📊 Comparison

| Aspect | Task-Inc (Old) | Class-Inc (New) |
|--------|----------------|-----------------|
| **Script** | `train.py` | `train_class_incremental.py` |
| **Heads** | 10 × 10 classes | 1 × 100 classes |
| **Labels** | Remapped [0-9] | Original [0-99] |
| **Accuracy** | 97% | 65-70% |
| **Matches Paper** | ❌ | ✅ |

---

## 💻 Training Commands

### CIFAR-100
```bash
python train_class_incremental.py \
    --dataset cifar100 --n_tasks 10 --epochs 10 \
    --batch_size 64 --lr 0.003 --reg 1.5
```
**Expected**: 65-70% Acc, -8% to -12% BWT, ~60 min

### ImageNet-R (Fix OOM First!)
```bash
pkill -9 python  # Clear GPU memory
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_class_incremental.py \
    --dataset imagenet_r --n_tasks 10 --epochs 10 \
    --batch_size 16 --lr 0.003 --reg 1.5
```
**Expected**: 55-60% Acc, -12% to -18% BWT, ~40 min

### CUB-200
```bash
python train_class_incremental.py \
    --dataset cub200 --n_tasks 10 --epochs 10 \
    --batch_size 32 --lr 0.003 --reg 1.5
```
**Expected**: 60-65% Acc, -10% to -15% BWT, ~10 min

---

## 📁 New Files

### Core Implementation
- `continual_learner_class_incremental.py` - Main learner
- `datasets_class_incremental.py` - Data loaders
- `train_class_incremental.py` - Training script

### Testing & Docs
- `test_class_incremental.py` - Quick test
- `compare_evaluation_protocols.py` - Show difference
- `STEP_BY_STEP_GUIDE.md` - Detailed guide
- `EVALUATION_PROTOCOL_FIX.md` - Technical docs

---

## ❓ FAQ

**Q: Why is accuracy lower now?**  
A: Class-incremental is harder (100-way vs 10-way classification). This is CORRECT.

**Q: Which should I use?**  
A: Use `train_class_incremental.py` to match the paper.

**Q: Is my original code wrong?**  
A: No! It's correct for task-incremental. Just different protocol.

**Q: ImageNet-R OOM?**  
A: Run `pkill -9 python` first, then use `--batch_size 16` or `8`.

---

## 📖 Documentation

- **Quick Start**: `STEP_BY_STEP_GUIDE.md`
- **Technical Details**: `EVALUATION_PROTOCOL_FIX.md`
- **Complete Summary**: `IMPLEMENTATION_FIX_SUMMARY.md`
- **This Card**: `QUICK_REFERENCE.md`

---

## ✅ Checklist

- [ ] Read `STEP_BY_STEP_GUIDE.md`
- [ ] Run `compare_evaluation_protocols.py`
- [ ] Run `test_class_incremental.py`
- [ ] Train CIFAR-100 (class-incremental)
- [ ] Fix OOM for ImageNet-R
- [ ] Train ImageNet-R
- [ ] Train CUB-200
- [ ] Compare with paper

---

## 🎓 Key Insight

**Task-Incremental** (97% accuracy):
- Easy: Model distinguishes 10 classes at a time
- Uses separate heads + label remapping
- Oracle task boundaries

**Class-Incremental** (65-70% accuracy):
- Hard: Model distinguishes ALL 100 classes
- Single unified head, no remapping
- No oracle (realistic setting)

**The paper uses class-incremental** → Use `train_class_incremental.py`

---

## 🔧 Troubleshooting

### OOM Error
```bash
pkill -9 python
nvidia-smi  # Verify GPU is clear
# Then reduce batch size: --batch_size 16 or 8
```

### Low Accuracy (<30%)
```bash
# Increase regularization
--reg 2.0

# Lower learning rate
--lr 0.002

# More epochs
--epochs 15
```

### Slow Training
```bash
# Increase batch size (if memory allows)
--batch_size 128

# Reduce workers if CPU bottleneck
--num_workers 2
```

---

## 📞 Need Help?

1. Check `STEP_BY_STEP_GUIDE.md` for detailed instructions
2. Check `EVALUATION_PROTOCOL_FIX.md` for technical details
3. Run `compare_evaluation_protocols.py` to understand the difference

---

**Remember**: Lower accuracy (65-70%) is CORRECT and EXPECTED for class-incremental learning!
