# PrettyCNN

A simple wrapper around the TensorFlow API to help you create beautiful Convolutional Neural Networks with ease.

### Note
Still under development and will be updated frequently.

### Todo
* Save the model
* More evaluation metrics
* Weight illustrator

# Example

You can start by defining the format of the image data you have in a Data object.
```python
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

data = Data(
    trainX=mnist.train.images,
    trainY=mnist.train.labels,
    testX=mnist.test.images,
    testY=mnist.test.labels, 
    img_size=28, 
    channels=1
)
```
Hereafter, you can define your model. No need to juggle with how weights should be shaped, it is all taken care of when passing the Data object.

```python
model = ModelBuilder(data).\
        input().\
        conv().\
        pool().\
        conv(64).\
        pool().\
        reshape().\
        dense().\
        dropout().\
        logits()
```
Hereafter we define how we want to evaluate our model. The Evaluator object will also contain all interesting information from the training one is is completed.

```python
evaluator = Evaluator(model).\
            softmaxCrossEntropy().\
            AdamOptimize()
```
Lastly we have to train our model. Here we can specify all the hyper-parameters.

```python
with tf.Session() as sess:

    Session(sess, evaluator).\
        steps(50).\
        rate(0.01).\
        batch(64).\
        dropout(0.4).\
        statusEvery(steps=5).\
        train().\
        test()
```

That's it! 