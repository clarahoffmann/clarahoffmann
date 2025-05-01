---
layout: post
title:  "Do you have what it takes to be Bayesian?"
date:   2023-01-06 13:36:25 +0100
categories: bayesian
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

Bayesian Statistics has become a trendy topic recently, especially in Deep Learning. This is no wonder, as Bayesian statistics offers a mathematically principled way to qunatify uncertainty. It allows to introduce prior knowledge into models, establish guarantees on convergence, produce flexible posterior predictive distributions and compute reliable prediction intervals. For neural networks, it combats overconfidence on out-of-distribution data, improves predictive performance and allows to impose statistical properties such as calibration or stationarity. \
While I have been working with Bayesian Statistics for a while now, I have to admit that it was more out of convenience and interest in mathematics, rather than being convinced of the Bayesian idea.

All Bayesian statistics is rooted in Bayes' rule, which combines prior information with observed data. In Deep Learning, it is common to formulate prior beliefs over a parameter \\ \theta \\ which is used in a model \\ f: x \rightarrow y \\ that maps some inputs \\ x \\ to some output \\y\\. (There are many more flavors as to what \\f\\ can constitute, but such a regression model is -- for me at least -- the easiest case).

Then Bayes rule tells us what the true distribution over \\ \theta \\ probably looks like given our prior beliefs and the data we observed.
\\ p(\theta | \mathcal{D}) = \frac{p(\theta)p(\mathcal{D} | \theta)}{p(\mathcal{D})} \propto p(\theta)p(\mathcal{D} | \theta)\\
where we call \\p(\theta | \mathcal{D}) \\ the *posterior*, \\ p(\theta) \\ the *prior*, \\p(\mathcal{D} | \theta)\\ the *likelihood* and \\p(\mathcal{D}) \\ the *evidence*


<p style="text-align: left;">
<img  style="display: block;  auto;"  src="https://clarahoffmann.github.io/clarahoffmann/images/bayes/bayes_casting.png" alt="drawing" style="width:500px;" >
</p>