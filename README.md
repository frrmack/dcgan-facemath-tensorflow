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

+ My extensions here are involve projecting face images onto the
  uniform z-space of the generator in a DCGAN trained on faces. To get
  realistic looking faces. I use the loss functions defined by Raymond
  Yeh et al. in their paper; without a mask, of course, since we are
  not completing images with missing pieces -- we have a fully complete real image, we are simply trying to find
  the best counterpart that the generator can generate. By using a
  perceptual loss in addition to the contextual one, we can find
  realistically closer looking projections by giving up some pixel
  level fidelity. Once we have these projections, they can be
  represented by linear vectors on a uniform space, meaning we can do
  linear algebra with them. This project aims to explore such
  operations.
