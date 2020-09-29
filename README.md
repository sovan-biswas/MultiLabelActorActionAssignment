# Discovering Multi-Label Actor-Action Association in a Weakly Supervised Setting
This repository provides a PyTorch implementation of the actor action assignment given label logits [Discovering Multi-Label Actor-Action Association in a Weakly Supervised Setting]().

### Qualitative Results:

<div align="center">
  <a href="https://github.com/sovan-biswas/MultiLabelActorActionAssignment/blob/master/sample_result/sample_results.avi"><img src="https://github.com/sovan-biswas/MultiLabelActorActionAssignment/blob/master/sample_result/sample_result.jpg" alt="IMAGE ALT TEXT"></a>
</div>

### Prerequisite
- Python >= 3.6 or 3.7
- PyTorch >= 1.3
- Numpy >= 1.18.0
- Cplex >= 12.10
- pickle >= 4.0 (optional)


### Tested with:
- Python >= 3.6
- PyTorch 1.4
- Numpy >= 1.19.0
- Cplex >= 12.10

### Usage Details
```shell
from LabelSpaceAssignment import LabelSpaceAssignment
logits_a = torch.randn(5,7) ## No. of instances = 5 and No. of classes = 7
bag_label = torch.empty(1,7).random_(2) ## No. of bags = 1 and No. of classes = 7
labelSpace = LabelSpaceAssignment(7) ## No. of classes = 7
labels,scores,omega_indx = labelSpace.assignmentSingle(logits_a,bag_label)
```

### Citation:

If you use the code, please cite

    S. Biswas and J. Gall.
    Discovering Multi-Label Actor-Action Association in a Weakly Supervised Setting.
    In Asian Conference on Computer Vision (ACCV), 2020
