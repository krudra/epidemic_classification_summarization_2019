How to run: python umls_disease_classifier.py <umls concept encoded file> <pre-trained model> <output file name>

Sample run: python umls_disease_classifier.py sample_input.txt UMLS_DISEASE_CLASSIFIER.pkl sample_output.txt

Note:
1. Set DICTIONARY and Tagger path accordingly
2. In general, Bag-of-Words model performs well in the in-domain setting. Hence, if we have annotated data for current epidemic, it is better to go with that dataset to train the model. However, in the absence of annotated data, this pretrained model may be used to classify tweets of current epidemic.
