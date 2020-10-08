# EBM+: Advancing Evidence-Based Medicine via two level automatic identification of Populations, Interventions, Outcomes in medical literature 

This repository hosts the implementation described in our paper [EBM+: Advancing Evidence-Based Medicine via two level automatic identification of Populations, Interventions, Outcomes in medical literature](https://www.sciencedirect.com/science/article/pii/S0933365720301986).
This repository is based on the [EBM-NLP](https://github.com/bepnye/EBM-NLP) as described in the publication 
[A Corpus with Multi-Level Annotations of Patients, Interventions and Outcomes to Support Language Processing for Medical Literature](https://arxiv.org/abs/1806.04185),
 which we are extending by the following contributions. Hence, similarities to the original code and repository are expected. 

In our work, we present  
1) A neural end-to-end PICO Entity Recognizer that identifies Population, Intervention/Comparator and Outcome entities in medical publications.
2) Novel Neural Network architecture for Entity Recognition involving a self-attention mechanism, a 2D Convolution feature extraction from character embeddings and a Highway residual connection.
3) A PICO Statement classifier that identifies sentences containing all the PICO Entities and answering clinical questions.
4) A high quality, manually annotated by medical practitioners dataset for PICO Statement classification.

The PICO Statements Dataset can be found [here](https://data.mendeley.com/datasets/p5rbn8mygp/1). 

# Requirements/Setup
The code runs under Python 3.6 or higher. The required packages are listed in the requirements.txt, which can be directly installed from the file:

```
pip install -r /path/to/requirements.txt
```
ELMo Weights and options files should be downloaded from [AllenNLP](https://allennlp.org/elmo).

 # Citation
 
If you find our work interesting, please cite using the following:
 
Stylianou, Nikolaos, et al. "[EBM+: Advancing Evidence-Based Medicine via two level automatic identification of Populations, Interventions, Outcomes in medical literature.](https://www.sciencedirect.com/science/article/pii/S0933365720301986)" Artificial Intelligence in Medicine 108 (2020): 101949.
