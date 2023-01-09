---
layout: post
title:  "Honey! I broke the Pytorch model - Debugging PyTorch models in a structured manner"
date:   2023-01-06 13:36:25 +0100
categories: pytorch python
---

When building custom PyTorch models, the model usually does not learn on the first try - even with extensive hyperparameter tuning. The same issue can appear after refactoring models that were already functional.
The reasons can be manifold: Most commonly, either 
1. 🪲 The model contains a bug - i.e. mixed-up channels, faulty augmentations, mixed up signs in a custom loss function etc.). This can easily happen when we're not dealing with a single off-the-shelf model but some sequences of models or custom loss functions, layers or add-ons that are taken freshly from the literature.
2. 💾 The data does not contain information to learn something (either due to wrong labels, mixing up datasets, etc.). This usually happens, when the data processing pipeline contains errors or we're dealing with a some prototyping project where success or failure is not clear yet.

# Find the culprit for the lack of convergence
<img align="right" src="../images/debug_data_seagull.jpg" alt="drawing" style="width:200px;"  >

Due to the plethora of potential reasons, debugging the training process is usually not straightforward. Diving into the debugger, decorating the code with print() statements, or going through the training examples by hand does not guarantee that the error is found - and can be quite overwhelming for engineers working under time pressure.
Luckily, there are more efficient approaches to locating the culprit for models that do not converge. This article introduces three main approaches to identify who's at fault for a lack of convergences and to prevent bugs from appearing again

1. **Synthetic data** Create synthetic data for your model - If the model doesn't learn with this data - there's likely a bug in the model. If it does learn with this data - your real-world data probably does not contain enough information to learn something!
2. **DL theory based checks**  Identify model bugs with simple checks. For example, The loss should continually decrease when training with one example only.
3. **Tests** Prevent model bugs with pytest. Setting up tests with synthetic data for training can prevent bugs are introduced during refactoring. Usually, it's enough to convert the checks from 2. into tests that are run during each commit. The tests can also be extended to check that model quality does not decrease with new updates.




# Creating synthetic training data for different input and label formats
Synthetic training data is a popular way to prove superiority of new DL methods with low computational effort. Creating synthetic training data requires obtaining some statistical knowledge. This extra effort makes some practicioners refrain from generating synthetic data. But once the concept for creating data is obtained once, the payoff is huge.


**Step 1:** Identify what label and input format is required
- *Do you have pixel level segmentation, object detection with bounding boxes, simple regression or classification?*

**Step 2:** Identify suitable and most basic distributional families that are required
- *For a classification scenario with 5 mutually exclusive labels, you'd like to select 5 normal distributions with as little overlap as possible.* (insert graph here)

**Step 3:** Generate your data and divide into train and test set

**Step 4:** Train your model
- If your model trains well after hyperparameter tuning: It's likely not buggy 🎉 -> check the validity your real-world data
- If it doesn't train: There's likely a 🪲 somewhere in the code...