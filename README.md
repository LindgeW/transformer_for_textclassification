# transformer_for_textclassification

## Transformer Architecture
<img src="imgs/trans_structure.png"/>

## Fine-tune Tricks
The demo apply transformer decoder to finish text classification task. Actually, it's hard to adjust the parameter value of model to proper one. Here, I draw a brief conclusion for how to effectively fine-tune the transformer encoder.

-- Adam optimization with learning rate of 1.5e-4 (1e-4 magnitude) and learning rate decay schedule with exponential decay. If lr is set larger(1e-3 magnitude), the result is not well.

-- For activation function, GeLU seems works worse than ReLU.

-- For regularization, dropout with a rate of 0.1 (but not in interval 0.2 to 0.5) and label smoothing for better generalization.


