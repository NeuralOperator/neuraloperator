.. _user_guide :
User Guide
===========

NeuralOperator provides all the tools you need 
to easily use, build and train neural operators for your own applications
and learn mapping between function spaces, in PyTorch.

~~~~~~~~~~~~

Intro to operator learning
----------------------------
To get a better feel for the theory behind our neural operator models, see :ref:`neuralop_intro`. 
Once you're comfortable with the concept of operator learning, check out specific details of our
Fourier Neural Operator (FNO) in :ref:`fno_intro`. Finally, to learn more about the model training
utilities we provide, check out :ref:`training_nos`.

~~~~~~~~~~~~

NeuralOperator library structure
---------------------------------

Here are the main components of the library:

================================= ================================
Module                             Description
================================= ================================
:mod:`neuralop`                   Main library 
:mod:`neuralop.models`            Full ready-to-use neural operators
:mod:`neuralop.layers`            Individual layers to build neural operators
:mod:`neuralop.data`              Convenience PyTorch data loaders for PDE datasets
:mod:`neuralop.training`          Utilities to train neural operators end-to-end
================================= ================================

The full API documentation is provided in the :ref:`api_ref`.

Finally, if you're building the library from source, your repository will also include the following directories:

================================= ================================
Directory                         Description
================================= ================================
:mod:`scripts`                    Training recipe scripts for our models on sample problems
:mod:`examples`                   More documented interactive examples, seen in :ref:`gallery_examples`
================================= ================================

~~~~~~~~~~~~

Interactive examples with code
----------------------------
We also provide interactive examples that show our library and neural operator models in action. 
To get up to speed on the code, and look through some interactive examples to help you hit the ground running,
check out our :ref:`gallery_examples`.

~~~~~~~~~~~~