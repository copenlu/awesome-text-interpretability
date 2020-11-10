# Opinion Papers

[The mythos of model interpretability](https://arxiv.org/pdf/1606.03490.pdf)

[Pathologies of Neural Models Make Interpretations Difficult](https://arxiv.org/pdf/1804.07781.pdf)
Model pathologies are due to model over-confidence or **second order sensitivity**. The first issue is addressed by training a classifier to be less certain about examples with reduced number of words. The second problem is interesting and understudied. There are only some indicators of the second order sensitivity, which in essense is common among interpretability techniques and occurse when slightly changing the heatmap of the input, which changes immensely the prediction of the interpretability technique. It would be interesting to study more which are the more stable interpretability techniques.

# Expalanation Studies in particular domains
## Social Sciences
* [Explanation in artificial intelligence: Insights from the social sciences](https://www.sciencedirect.com/science/article/pii/S0004370218305988?casa_token=WLyrfIWttxIAAAAA:kD_vnn8GYC76FXodoKLPqTZP8N3BP9VR9BSnYv5uybq3N3_WpdXRiUxJ2EvnF02FKgT7S37o0P8) Views from psychology and philosophy on explanations. Among many other things, authors point that explanations can have different forms, where the most useful are the contrastive explanations (can be 3 types: Why does object a have property P, rather than property Q?, Why does object a have property P, while object b has property Q?, Why does object a have property P at time t, but property Q at time t′)]

## Fact Checking
* [Explainable Fact Checking with Probabilistic Answer Set Programming, TTO 2019](https://truthandtrustonline.files.wordpress.com/2019/09/paper_15.pdf) - Retrieve tripes from knowledge graphs and combine them using rules to produce explanations for fact checking

## Machine Reaching Comprehension / Question Answering
* [Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering, EMNLP 2020](https://arxiv.org/abs/2010.03274) - 3 datasets and delexicalized chain representations in which repeated noun phrases are replaced by variables, thus turning them into generalized reasoning chains

# Explainability Techniques

## Saliency maps
* [Hierarchical interpretations for neural network predictions, ICLR 2019](https://openreview.net/forum?id=SkEqro0ctQ) - Provide a hierarchical visualisation of how words contribute to phrases and how phrases contribute to bigger piesces of text and eventually to the overall prediction.
* [Towards a Deep and Unified Understanding of Deep Neural Models in NLP, ICML 2019](https://www.microsoft.com/en-us/research/uploads/prod/2019/05/camera_paper_with_supp_3.pdf) Provide a technique to show how word saliency develops through all of the layers until the final output. The authors compare their approach to LRP, perturbations and gradients. They also provide a comparison between explanations from BERT, Transformer, LSTM, CNN.

## Generating Rationales 
* [Towards Explainable NLP: A Generative Explanation Framework for Text Classification, ACL 2019](https://arxiv.org/pdf/1811.00196.pdf)
* [Why do you think that? Exploring Faithful Sentence-Level Rationales Without Supervision, EMNLP 2020](https://arxiv.org/pdf/2010.03384.pdf ) - a differentiable training-framework to create models which output faithful rationales on a sentence level, by solely applying supervision on the target task; model solves the task based on each rationale individually and learns to assign high scores to those which solved the task best
* [F1 is Not Enough! Models and Evaluation Towards User-Centered Explainable Question Answering](https://arxiv.org/pdf/2010.06283.pdf) - two novel evaluation scores: (i) tracks prediction changes when removing facts, (ii) assesses whether the answer is contained in the explanation or not; Further strengthen the coupling of answer and explanation prediction in the model architecture and during training

# Improving Model Interpretability
* [Learning Variational Word Masks to Improve the Interpretability of Neural Text Classifiers, EMNLP 2020](https://arxiv.org/pdf/2010.00667.pdf)  - variational word masks (VMASK) that are inserted into a neural text classifier, after the word embedding layer, and trained jointly with the model. VMASK learns to restrict the information of globally irrelevant or noisy wordlevel features flowing to subsequent network layers, hence forcing the model to focus on important features to make predictions

## Other
* [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. EMNLP 2020](https://arxiv.org/abs/2009.10795) - a model-based tool to characterize and diagnose datasets

# Adversarial Attacks as Model Interpretations
* [More Bang for Your Buck: Natural Perturbation for Robust Question Answering, EMNLP 2020](https://arxiv.org/abs/2004.04849)
* [FIND: Human-in-the-Loop Debugging Deep Text Classifiers, EMNLP 2020](https://arxiv.org/abs/2010.04987)
* [Beat the AI: Investigating Adversarial Human Annotation for Reading Comprehension(UCL group), EMNLP 2020](https://arxiv.org/pdf/2002.00293.pdf)

# Evaluation
* [Sanity Checks for Saliency Maps](https://arxiv.org/pdf/1810.03292.pdf) Authors propose *model parameter and label randomization* to estimate the invariance of the interpretability tools to the model and the data. They find that guided BackProb and Grad CAM are invariant to both.

* [Manipulating and Measuring Model Interpretability](https://arxiv.org/pdf/1811.00196.pdf) The authors examine how number of features and transparency of model influence the model interpretability. They found that a smaller model was easier for simulation. However, showing a simpler model did not help the annotators to correct the model's behavior or identify wrong decisions. 

* [Human-grounded Evaluations of Explanation Methods for Text Classification, EMNLP 2019](https://arxiv.org/pdf/1908.11355.pdf) The authors  design three tasks, where humans evaluate the following explanation techniques: LIME, LRP, DeepLIFT, Grad-CAM-Text and Decision Trees (for words and n-grams). They find that LIME is the most class-discriminative approach. Unfortunately, the annotator agreement is considerably low in most tasks and one general improvement would be to provide the words and n-grams together with the context they appear in. 

* [Analysis Methods in Neural Language Processing: A Survey, TACL 2019](https://arxiv.org/pdf/1812.08951.pdf)

* [Interpretation of Neural Networks is Fragile](https://arxiv.org/pdf/1710.10547.pdf)
The authors devise adversarial attacks with different types of perturbations, which do not change the prediction or the confidence scores of the model, but change the explanation with a lot. This, however does not mean that the components that the different interpretation techniques use did not change. It might be a good idea to measure the change of the model's weights change, not only the confidence, because the model might pick up other words and still be confident (pathologies in neural networks). The most robust method turned to be the integrated gradients. The analysis is for images.


# Annotated Human Rationales and Datasets
## On Human Rationales
* [From Language to Language-ish: How Brain-Like is an LSTM's Representation of Nonsensical Language Stimuli?, EMNLP 2020](https://arxiv.org/pdf/2010.07435.pdf) - The syntactic signatures available in Sentence and Jabberwocky LSTM representations are similar, and can be predicted from either the Sentence or Jabberwocky EEG. From our results, we can infer which LSTM representations encode semantic and/or syntactic information. We confirm using syntactic and semantic probing tasks. Our results show that there are similarities between the way the brain and an LSTM represent stimuli from both the Sentence (within-distribution) and Jabberwocky (out-of-distribution) conditions.
* [Evaluating and Characterizing Human Rationales, EMNLP 2020](https://arxiv.org/pdf/2010.04736.pdf) - An open question, however, is how human rationales fare with these automatic metrics - do not necessarily perform well- reveal irrelevance and redundancy. Our work leads to actionable suggestions for evaluating and characterizing rationales. 

## Datasets with Highlights
* e-SNLI [e-SNLI: Natural Language Inference with Natural Language Explanations](https://arxiv.org/pdf/1812.01193.pdf)
* CoS-E [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/pdf/1906.02361.pdf)
* BeerAdvocate [Rationalizing Neural Predictions](https://arxiv.org/pdf/1606.04155.pdf)
* BabbleLabble [Training Classifiers with Natural Language Explanations](https://arxiv.org/pdf/1805.03818.pdf)

## Datasets with Textual explanations
* e-SNLI [e-SNLI: Natural Language Inference with Natural Language Explanations](https://arxiv.org/pdf/1812.01193.pdf)
* LIAR-PLUS [Where is your Evidence: Improving Fact-checking by Justification Modeling](https://www.aclweb.org/anthology/W18-5513.pdf)
* CoS-E [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/pdf/1906.02361.pdf)

# Demos
[AllenNLP INterpret](https://allennlp.org/interpret), [Demo](https://demo.allennlp.org/reading-comprehension)

# Tutorials
[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)

