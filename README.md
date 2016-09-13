# Face Math with Deep Learning in TensorFlow

+ This repository extends upon
  [Taehoon Kim's](http://carpedm20.github.io/) MIT licensed
  [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
  project, and [Brandon Amos's](https://bamos.github.io/)
  [dcgan-completion.tensorflow](https://github.com/bamos/dcgan-completion.tensorflow)
  project, which extended the former one and is also MIT
  licenced. This project is also MIT licensed.

+ The Tensorflow implementation of Deep Convolutional Generative
  Adversarial
  Networks([paper by Alec Ratford et al.](https://arxiv.org/abs/1511.06434))
  is done by Taehoon Kim. DCGANs are a specific implementation of
  adversarial training
  ([original GAN paper by Ian Goodfellow et al.](https://arxiv.org/abs/1406.2661))
  with a specific architecture are training approaches.

+ Brandon Amos has taken Taehoon Kim's code and extended it to
  implement image completion following the
  [Semantic Image Inpainting with Perceptual and Contextual Losses](https://arxiv.org/abs/1607.07539)
  by Raymond Yeh et al. He has written a nice [blog post about this implementation including many details](http://bamos.github.io/2016/08/09/deep-completion/).

+ **My extensions here involve testing if the GANs can be used like encoders.**

+ **There are two majorly explored ways of generating images with deep learning.** One is Variational Autoencoders (**VAE**s) and the other is Generative Adversarial Networks (**GAN**s). The idea behind VAEs is to teach a neural network framework (two networks, one encoder that converts an image into a single vector of numbers, one decoder that takes this vector and converts it back to an image, trying to get as close as possible to the original) to represent a real image by a vector of latent variables. This is very useful, since it can allow linear arithmetic on images, directly manipulating facial features, such as making a person smile, turn their head, wear glasses, by simply manipulating the latent representation vector. GANs, on the other hand, do not directly train to find the best representation for a given real image. They try to be the best fake image generator. They achieve this through an arms race between two networks. One's (discriminator) is constantly trying to get better at distingushing real images from fake ones, the other (generator) is constantly trying to fool the first one by converting vectors of numbers into more and more realistic looking images. 

+ **GANs combined with convolutional artichectures have been shown to generate amazingly realistic images.** They have been very successful in many tasks related to generating images. I was curious if they could work by themselves as a sorts of encoder as well. **For GANs, there is of course no direct way to extract which vector of numbers a real image would correspond to** (unlike VAEs, where you can just put the image into the encoder and get the related vector). The GAN generator *takes* a representation vector z and converts it into a real image. Not the other way around.

+ **However, one way** of finding how a GAN generator would perhaps think of a given image as a vector of numbers _(as a point in z space, rather than the pixel matrix space)_ **is to search for the z vector that results in the image that is "closest" to the real image.** We can do that by defining a loss function that measures the difference between the original image and the generated image for a given z. Then we can search for the z vector that minimizes this loss. (This would mean projecting an original image onto the z-space of the generator.)
 
+ **How do we measure the difference between the generated and original image?** A straightforward loss function is what Raymond Yeh et al. call a contextual loss, which is just the sum of absolute differences between pixel values of the original and the generated image. However, in their work to complete missing parts of given images, they realized that this loss by itself can lead to not-so-realistic looking results. GANs already have a real/fake discriminator network, ready to inform us if a given generated image looks right or not. So instead using only contextual loss and saying "find me the z that gives the closest pixel-to-pixel image from the generator", **we can say "find me a z that gives a really close pixel-to-pixel image that also looks realistic."** The realistic part is called the perceptual loss, and it is basically how far away from the reality the generated image looks according to our discriminator. We combine these two and say thwe total loss is contextual loss plus _lambda_ times perceptual loss. _lambda_ here is a parameter we can turn like a knob to add more weight to one of these losses.

+ Once we have these projections, they can be
  represented by linear vectors on a uniform space, meaning we can do
  linear algebra with them. **Will these projections resemble the original pictures somewhat faithfully? Will the representation vectors and their manipulations work worse than, as well as, or better than their VAE counterparts?** If they behave differently, how so? This project aims to explore such
  questions.

