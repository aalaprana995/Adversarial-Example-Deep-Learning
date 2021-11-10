# Adversarial-Example-Deep-Learning
This project provides insight into what is Adversarial Examples and what impact it has on deep learning models. 

# Deep Fool 

- It is a method to generates minimum perturbation which when added to the original images creates a new images that visiually looks similar to the original image but fools the model into believing that it belongs to a different class.
- This method produces adversarial examples with minimum variation in the image.

## Intution
The intution for the working of Deep fool can be understood using the figure below.\
Let us have a classifier "f" and  data point in this case a image it classifies be "x0" and with label "0".\
If we want to change the label to "1" instade of "0" for the image "x0" we need to shift the point "x0" to the other side of the classifier.\
Thus the robustness of classifier for the image/ data point is equal to the distance of the data point from the classifier.\
The minimum value that we can add to the "x0" is equal to the perpendicular distance "r" of the data point to the classifier.\
**Note** \
We sometimes might end up on the classifier plane. Thus to avoid such situation we add a small value let it be called "n" to the calculated purturbation which ranges form 0 to 1. 

![imshs](https://user-images.githubusercontent.com/93336207/140392412-aee62e0b-b09f-4f87-b595-2dfe93cb8492.png)

## Algorithm 
## Binary class classifier
![deep](https://user-images.githubusercontent.com/93336207/140391094-eea5c7d9-a13c-4cc1-a1bb-3e0a7c98717f.png)

## Multiclass classifier
Multi class classifier is similar to binary classifier, but in addition to calculating the mininmum perpendicular distance we also select the nearest hyper-plane/ classifier.\
This works as Deep Fool is not a **"Targeted Adversarial Generator"** method.\
![multi-class-deep](https://user-images.githubusercontent.com/93336207/140391103-5ad891c4-6c71-417a-9795-22821a0aecda.png)

## Fast Gradient Sign Method 

## Intution
- A simple linear model can be described as (W)T * x, where W is the weight matrix and x is the input. 
- The input is made up of many features, and the precision of any given particular feature is limited.
- Now let’s add a small noise η, such that η < ϵ (ϵ is smaller than the precision of the features) to every feature of x. We can call this new input x̄. 
```
x̄ = x + η
```
We can write the dot product between the weight matrix W and x̄ as
```
(W)T * x̄ = (W)T * x + (W)T * η
```
- This means that the activation of the network increases by (W)T * η. 
- Such small changes in activation can grow linearly with the increase in dimensions. Hence, for high-dimensional problems, we can make small changes to the input that can add up to one big change to the output. Thus, even a smaller model is vulnerable to adversarial examples, provided the input is high dimensional.
### Fast Gradient Sign Method 
The idea behind FGSM is surprisingly simple: we do opposite of the typical gradient descent in order to maximize the loss, since confusing the model is the end goal of adversarial attack.
Therefore, we consider x, the model’s input, to be a trainable parameter. Then, we add the gradient to its original input variable to create a perturbation. Mathematically, this can be expressed as follows:
η=ϵ⋅sign(∇xJ(w,x,y))
where J represents the cost function. 
Then, we can create an adversarial example via
x~=x+ϵ⋅sign(∇xJ(w,x,y))

This is the crux of the fast gradient sign method: we use the sign of the gradient, multiply it by some small value, and add that perturbation to the original input to create an adversarial example.

One way to look at this is in terms of first-order approximation. Recall that
f(x~)≃f(x)+(x~−x)⊤∇xf(x)

In this context, we can consider f
to be the cost function J

, which then turns into
J(w,x~)=J(x,w)+(x~−x)⊤∇xJ(w,x)

Then, the goal of an adversarial attack is to maximize the second term in the addition. Since there is an infinity norm constraint on the perturbation, namely
∥x~−x∥∞≤ϵ

with some thinking we can convince ourselves that the perturbed example that maximizes the loss function is given by
x~=x+ϵ⋅sign(∇xJ(w,x,y))
