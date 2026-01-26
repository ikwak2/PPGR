## Experiment Results
(30 epoch로 통일)
______________________________________________________
### ResNet Encoder + Element-wise Sum Concat Fusion + Cross-Attention
- Best Validation Accuracy : 93.5%
- EER (Equal Error Rate): 3.0167%

### ResNet Encoder (Large) + Element-wise Sum Concat Fusion + Cross-Attention
ResNet의 Layer를 추가함 (더 깊게)
- Best Validation Accuracy : 95.2%
- EER (Equal Error Rate): 4.1005%

### ResNet Encoder (Large) + Concat Fusion + Cross-Attention
- Best Validation Accuracy : 94.6%
- EER (Equal Error Rate): 2.0278%

### ResNet Encoder (Large) + SE Layer + Concat Fusion + Cross-Attention
- Best Validation Accuracy : 94.6%
- EER (Equal Error Rate): 1.5602%



