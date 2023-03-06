---
layout: post
title:  "Honey! I broke the Pytorch model - Debugging PyTorch models in a structured manner"
date:   2023-01-06 13:36:25 +0100
categories: pytorch python
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

When building custom PyTorch models, the model usually does not learn on the first try - even with extensive hyperparameter tuning. The same issue can appear after refactoring models that were already functional.
The reasons can be manifold: Most commonly, we are dealing with one (or both) of two problems:

<table border="0">
 <tr>
    <td><b style="font-size:30px">🪲 Model bugs</b></td>
    <td><b style="font-size:30px">💾 Data issues</b></td>
 </tr>
 <tr>
    <td>The model contains a bug - i.e. mixed-up channels, faulty augmentations, mixed up signs in a custom loss function etc.). This can easily happen when we're not dealing with a single off-the-shelf model but some sequences of models or custom loss functions, layers or add-ons that are taken freshly from the literature. ...</td>
    <td>The data does not contain information to learn something (either due to wrong labels, mixing up datasets, etc.). This usually happens, when the data processing pipeline contains errors or we're dealing with a some prototyping project where success or failure is not clear yet.</td>
 </tr>
</table>


# Find the culprit for the lack of convergence
<img align="right" src="https://clarahoffmann.github.io/clarahoffmann/images/debug_data_seagull.jpg" alt="drawing" style="width:270px;"  >

Due to the plethora of potential reasons, debugging the training process is usually not straightforward. Diving into the debugger, decorating the code with print() statements, or going through the training examples by hand does not guarantee that the error is found - and can be quite overwhelming for engineers working under time pressure.
Luckily, there are more efficient approaches to locating the culprit for models that do not converge. This article introduces three main approaches to identify who's at fault for a lack of convergences and to prevent bugs from appearing again

1. **Synthetic data** Create synthetic data for your model - If the model doesn't learn with this data - there's likely a bug in the model. If it does learn with this data - your real-world data probably does not contain enough information to learn something!

2. **DL theory based checks**  Identify model bugs with simple checks. For example, The loss should continually decrease when training with one example only.

3. **Tests** Prevent model bugs with pytest. Setting up tests with synthetic data for training can prevent bugs are introduced during refactoring. Usually, it's enough to convert the checks from 2. into tests that are run during each commit. The tests can also be extended to check that model quality does not decrease with new updates.




# Creating synthetic training data for different input and label formats
Synthetic training data is a popular way to prove superiority of new DL methods with low computational effort. Creating synthetic training data requires obtaining some statistical knowledge. This extra effort makes some practicioners refrain from generating synthetic data. But once the concept for creating data is obtained once, the payoff is huge.

<p style="text-align: center;">
<img  style="display: block;  auto;"  src="https://clarahoffmann.github.io/clarahoffmann/images/synthetic_data_process.png" alt="drawing" style="width:500px;" >
</p>

Step (1) comprises questions such as *Do you have pixel level segmentation, object detection with bounding boxes, simple regression or classification?*. Step (2) is the most challenging and is described later in this article.
Interpreting the results from our training run is straightforward:
- If your model trains well after hyperparameter tuning: It's likely not buggy 🎉 -> check the validity your real-world data
- If it doesn't train: There's likely a 🪲 somewhere in the code...


## How to generate synthetic training data
There are several ways to generate synthetic training data. Luckily, for model debugging we can stick to very simple data generating mechanisms! We just want to test whether our model is working or not.
Let's distinguish between different application cases:

1. **Regression:**  

**Synthetic data creation by hand:**  
Synthetic data for regression cases needs to be nonlinear. The most straightforward way to generate nonlinear data is to leverage sin and cosine functions.
Input datapoints can be drawn from simple, cheap distributions such as the uniform distribution.
Draw \\({x_i}_{i=1}^n\\) from \\(\mathcal{U}(-1,1) \\), generate \\(y_i = \sin(8*(x_i - 1.2)) + \epsilon_i \\), where  \\(\epsilon_i \sim \mathcal{N}(0, 0.01^2) \\).

Dataset split: Train and test splits can be created in a standard way by randomly assigning x% of the data to the train set and (100 - x)% to the test set. If out-of-distribution samples are expected and the model should be robust against this, a gap split is more suitable. Here, the input space is partitioned into two disjoint sets. For example, if \\X \sim (\mathcal{U}(-1,1) \\), we choose the train set as all \\(x_i \in [-0.8, 0.8]\\) and the test set as \\(x_i \in [1, -0.8) \cup (0.8 1] \\). This way we achieve a 80/20 train/test split.


**Off-the-shelf synthetic data in Python:**  

Several packages in Python provide functionalities to generate synthetic, nonlinear data.


2. **Classification:** If the number of classes are few, we can generate inputs for each class from different normal distributions, which do not have a great overlap in their densities. Generate from \\(X_i\\) corresponding to class \\(i\\) via \\(X_i \sim N(\mu, \Sigma)\\) and save \\(\{X_i, i\}\\) as one training example. Repeat this as many times as you need samples. This approach also allows to directly determine whether you want to test on an imbalanced or balanced dataset. In a more advanced version, it would also be possible to generate from a regression model as in 1. and use a link function as in generalized linear models.

3. **Object detection (2D or 3D tensors):**
- Create empty patches (all initialized to zero) and sample random bounding box coordinates on the patch. Set all values in the patch to a distinct value for each object class. For further complexity, a value for each patch can be generated from a regression model.

<p align="center">
<img  src="https://clarahoffmann.github.io/clarahoffmann/images/object_detection_generate_data.png" alt="drawing" style="width:500px;"  >
</p>


# Theory-based checks
Training PyTorch models typically involves three building blocks:
(1) model structure
(2) train and validation methods
(3) data loaders 

<p align="center">
<img  src="https://clarahoffmann.github.io/clarahoffmann/images/building_blocks.png" alt="drawing" style="width:700px;"  >
</p>

Each one of these can be subject to errors. It can be quite tricky to locate errors in these structures. This is especially true if we're dealing with dimensionality, value, or range errors. These types of errors often don't hinder our model from running and still produce reasonable outputs.

*The term "theory-based" might strike experienced users as excessive here - given the simple nature of the checks. For example, checking whether weights are updated in a training loop might seem natural. However, also this simple check is based on DL theory and our model construct. For example, we might be doing a fine-tuning task, where only the weights of the last layer should be updated. Checking that all previous layers are frozen and the last one is updated is completely rooted in theory in this case".*

Here is a short summary of the most common model bugs that can appear in the basic building blocks

<table border="0">
 <tr>
    <td><b style="font-size:30px">model structure</b></td>
    <td><b style="font-size:30px">training/validation loop </b></td>
    <td><b style="font-size:30px">dataloader</b></td>
 </tr>
 <tr>
    <td>
    <ul>
    <li>Weights not updated in correct manner</li>
    <li>Layers skipped or not connected</li>
    <li>Activation functions missing</li>
   </ul>
   </td>
<td>
<ul>
    <li>NaN/Inf values appearing during forward/backward passes</li>
    <li>Incorrect output range (e.g. softmax outputs not within [0,1] range </li>
    <li>Activation functions missing</li>
   </ul>
    </td>
    <td>
    <ul>
    <li>Wrong format of data</li>
    <li>Input channels mixed-up </li>
    <li>Incorrect input value range (e.g. images not normalized to [0,1])</li>
    </ul>
     </td>
 </tr>
</table>