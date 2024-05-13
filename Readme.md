# A simple project to ilustrate Neural Networks

Features:
- Tensorflow version
- Creating tensors
- Arithmethic Operations
- Algebra Linear
- Tensor data types
- Linear Equation Solution
- Reduction
- Transformation
- Gradient
- Decorator
- Neural Network concept
    - Mathematical concept of a Neuron
    - Basic structure of the multileyer perceptron (MLP)
    - Forward Propagation
    - Backward Propagation
    - Wine prediction model using the created neural network
    - Wine prediction model using sklearn
    - Housing price prediction model using sklearn

## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone https://github.com/jacesca/StatisticsConcepts.git
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```

## Run ML model
```
python TensorBasics.py
python TensorDataTypes.py
python TensorConstantsAndVariables.py
python TensorArimethics.py
python TensorAlgebraLinear.py
python TensorLinealEqSolution.py
python TensorFirstNeurolNet.py
python TensorTransformations.py
python TensorDatasetFixEx.py
python TensorReduction.py
python TensorWeatherCaseEx.py
python TensorGradient.py
python TensorDecorator.py
python TensorXORImplementation.py
```
Related to Concept
```
python ConceptPersonalNeuron.py         # Neuron class implementation
python ConceptPersonalMultilayer.py     # Multilayer class implemnetation
python ConceptPersonalForward.py        # Forward Propagation method implementation
python ConceptPersonalBackward.py       # Backward Progagation method implementation
python WineForecastingPersonalNeuron.py # Problem solved with the implemented class
python WineForecastingSklearn.py        # Problem solved using sklearn neural network
```

## Others
- Proyect in GitHub: https://github.com/jacesca/StatisticsConcepts
- Commands to save the environment requirements:
```
conda list -e > requirements.txt
# or
pip freeze > requirements.txt

conda env export > flask_env.yml
```
- For coding style
```
black model.py
flake8 model.py
```

## Neural Networks or Traditional Models
Differences
- Complexity
    - Traditional Models: Typically less complex, easier to visualize, and more interpretable. Think of a simple line in linear regression or the branches of a decision tree.
    - Neural Networks: Can be highly complex, especially deep neural networks. They can have millions of parameters, making them a sort of "black box" where it's hard to discern exactly how they're making decisions.
- Training Time
    - Traditional Models: Generally faster to train because of their simplicity.
    - Neural Networks: Require more computational power and time, especially when dealing with a vast amount of data or deeper architectures.
- Data Requirements
    - Traditional Models: Can work well with smaller datasets than neural networks.
    - Neural Networks: Often require larger datasets to generalize well and avoid overfitting.

Limitations
- Traditional Models
    - Linearity: Some models, like linear regression, assume a linear relationship between features and output.
    - Feature Engineering: Require manual intervention and domain knowledge to create and select the right features.
    - Scalability: While they can handle large datasets, they might not capture complex patterns as effectively as neural networks.
Neural Networks
    - Overfitting: Without proper techniques like regularization, they can overfit on training data, leading to poor generalization.
    - Interpretability: Often considered "black boxes", making it hard to explain their decisions.
    - Computational Needs: Require more computational resources, especially deep neural networks.
    - Feature Engineering: They also require feature engineering like traditional models, but neural networks are less sensitive to this, because they can filter out unnecessary features during training.

How to Choose Between Them
- Dataset Size: For smaller datasets, traditional models might be more suitable, while larger datasets might benefit from neural networks.
- Complexity of a Problem: For simpler patterns, a traditional model might suffice. But for more complex patterns, like image recognition, a neural network might be necessary.
- Interpretability: If you need to explain your model's decisions, traditional models are usually more interpretable.
- Resources: If computational resources or training time are a concern, traditional models might be a better starting point.

Some Neural Network types:
- Feedforward Neural Networks (FNN) or Multi-layer Perceptrons (MLP)
> This is a classic NN architecture, a direct extension of the single-layer perceptron to multiple layers. These are the foundational architectures upon which most other neural network types are built. It is the architecture that we have considered in this course.

- Convolutional Neural Networks (CNN)
> CNNs are especially powerful for tasks like image processing (problems such as image classification, image segmentation, etc.) because they're designed to automatically and adaptively learn spatial hierarchies of features.

- Recurrent Neural Networks (RNN)
> RNNs have loops to allow information persistence. Unlike feedforward neural networks, RNNs can use their internal state (memory) to process sequences of inputs, making them extremely useful for time series or sequential data. They're broadly used for sequence prediction problems, like natural language processing or speech recognition.
> - Long Short-Term Memory (LSTM): Overcomes the vanishing gradient problem of RNNs, making it easier to learn from long-term dependencies.
> - Gated Recurrent Units (GRU): A simpler and more efficient variant of LSTM. However, it learns complex patterns in the data worse than LSTM.

- Autoencoders (AE)
- Generative Adversarial Networks (GAN)
- Modular Neural Networks (MNN)

Libraries for Deep Learning
- Training deep neural networks requires more than the classic machine learning library scikit-learn offers. The most commonly used libraries for working with deep neural networks are TensorFlow and PyTorch. Here are the main reasons why they are preferred for this task:
    - Performance and Scalability: TensorFlow and PyTorch are designed specifically for training models on large amounts of data and can run efficiently on graphics processing units (GPUs), which speeds up training.
    - Flexibility: Unlike scikit-learn, TensorFlow and PyTorch allow you to create arbitrary neural network architectures, including recurrent, convolutional, and transformer structures.
    - Automatic Differentiation: One of the key features of these libraries is the ability to automatically compute gradients, which is essential for optimizing weights in neural networks.

## Extra documentation
- [Disable Tensorflow debugging information](https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
- [How to disable ConvergenceWarning using sklearn?](https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn)
- [This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX AVX2. To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.](https://stackoverflow.com/questions/65298241/what-does-this-tensorflow-message-mean-any-side-effect-was-the-installation-su)
