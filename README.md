Interpretability in NLP

# On interpretability
[The mythos of model interpretability](https://arxiv.org/pdf/1606.03490.pdf)

[Pathologies of Neural Models Make Interpretations Difficult](https://arxiv.org/pdf/1804.07781.pdf)

# Evaluation of interpretability techniques
* [Manipulating and Measuring Model Interpretability](https://arxiv.org/pdf/1811.00196.pdf) The authors examine how number of features and transparency of model influence the model interpretability. They found that a smaller model was easier for simulation. However, showing a simpler model did not help the annotators to correct the model's behavior or identify wrong decisions. 

* [Human-grounded Evaluations of Explanation Methods for Text Classification, EMNLP 2019](https://arxiv.org/pdf/1908.11355.pdf) The authors  design three tasks, where humans evaluate the following explanation techniques: LIME, LRP, DeepLIFT, Grad-CAM-Text and Decision Trees (for words and n-grams). They find that LIME is the most class-discriminative approach. Unfortunately, the annotator agreement is considerably low in most tasks and one general improvement would be to provide the words and n-grams together with the context they appear in. 

* [Analysis Methods in Neural Language Processing: A Survey, TACL 2019](https://arxiv.org/pdf/1812.08951.pdf)

# Interpretability Techniques
* [Hierarchical interpretations for neural network predictions, ICLR 2019](https://openreview.net/forum?id=SkEqro0ctQ) - Provide a hierarchical visualisation of how words contribute to phrases and how phrases contribute to bigger piesces of text and eventually to the overall prediction.
* [Towards a Deep and Unified Understanding of Deep Neural Models in NLP, ICML 2019](https://www.microsoft.com/en-us/research/uploads/prod/2019/05/camera_paper_with_supp_3.pdf) Provide a technique to show how word saliency develops through all of the layers until the final output. The authors compare their approach to LRP, perturbations and gradients. They also provide a comparison between explanations from BERT, Transformer, LSTM, CNN.
* [Towards Explainable NLP: A Generative Explanation Framework for Text Classification, ACL 2019](https://arxiv.org/pdf/1811.00196.pdf)

# Datasets with annotated explanations

## Highlights
* e-SNLI [e-SNLI: Natural Language Inference with Natural Language Explanations](https://arxiv.org/pdf/1812.01193.pdf)
* CoS-E [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/pdf/1906.02361.pdf)
* BeerAdvocate [Rationalizing Neural Predictions](https://arxiv.org/pdf/1606.04155.pdf)
* BabbleLabble [Training Classifiers with Natural Language Explanations](https://arxiv.org/pdf/1805.03818.pdf)

## Textual explanations
* e-SNLI [e-SNLI: Natural Language Inference with Natural Language Explanations](https://arxiv.org/pdf/1812.01193.pdf)
* LIAR-PLUS [Where is your Evidence: Improving Fact-checking by Justification Modeling](https://www.aclweb.org/anthology/W18-5513.pdf)
* CoS-E [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/pdf/1906.02361.pdf)

# Models


# Use cases

# Demos
[AllenNLP INterpret](https://allennlp.org/interpret), [Demo](https://demo.allennlp.org/reading-comprehension)

# Tutorials
[Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)

