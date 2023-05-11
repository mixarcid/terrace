.. _Modules:

Modules
===============

.. jupyter-execute::
    :hide-code:
    
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))

.. +++++++++++++++
.. Usage
.. +++++++++++++++

Terrace's ``Module`` class is a simpler, faster way of defining PyTorch 
modules. Instead of replicating the structure of your neural network 
in both the ``__init__`` and ``forward`` methods, you can define 
everything in the ``forward`` method. All the modules are created 
the first time you call the method, and are re-used afterward.

Here's an example of a basic neural network with one hidden layer:

.. jupyter-execute::

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import terrace as ter

    class BasicNN(ter.Module):

        def forward(self, x):
            # we always need to call start_forward at the beginning
            # of the method, so the module knows to reset all the
            # internal counters
            self.start_forward()

            # Pytorch >= 1.13 also has a LazyLinear class
            # Note that I'm only defining the output dimension;
            # the input dimension is automatically determined
            hid = F.relu(self.make(ter.LazyLinear, 100)(x))
            out = self.make(ter.LazyLinear, 1)(hid)
            return out
        
    x = torch.randn(16, 128)
    model = BasicNN()
    out = model(x)

    print(out.shape)

The key idea is to use the ``make`` method to both instantiate and run
submodules. During the first ``forward`` call, calling
``make(ModuleType, *args, **kwargs)`` will instantiate a submodule
``ModuleType(*args, **kwargs)``. When we call ``forward`` again,
``make`` will simply return the same module it gave us before.

In the above example, the reason we're using the ``LazyLinear`` module
instead of ``nn.Linear`` is that ``LazyLinear`` will infer the input
dimension automatically. But how? It turns out we already have all the
tools we need to easily implement the ``LazyLinear`` class ourselves. Here's 
how we could do it with only a couple lines:

.. jupyter-execute::

    class MyLazyLinear(ter.Module):

        def __init__(self, out_feat):
            super().__init__()
            self.out_feat = out_feat

        def forward(self, x):
            self.start_forward()
            in_feat = x.size(-1)
            return self.make(nn.Linear, in_feat, self.out_feat)(x)
        
    linear = MyLazyLinear(16)
    print(linear(x).shape)

Terrace provides a couple dimension-inferring wrappers around PyTorch's
modules, including ``LazyLayerNorm`` and ``LazyMultiheadAttention``, but
nothing remotely comprehensive. However, as you can see, it is very easy to
create your own wrappers.

+++++++++++++++
Gotchyas
+++++++++++++++

While terrace's ``Module`` is more convenient in most cases, there are
some gotchyas you'll need to consider:

First, and most importantly, data-dependant control flow will not work
out of the box, and (very unfortunately) will sometimes fail silently.

For instance, the following model will not work:

.. jupyter-execute::

    class BadNetwork1(ter.Module):

        def forward(self, x):
            self.start_forward()
            h = self.make(ter.LazyLinear, 10)
            # the number of times this loop
            # get executed is data-dependant
            for n in range(int(torch.amax(x))):
                h = self.make(ter.LazyLinear, 10)(x)

When we run this model the first time, it will work fine. However, as soon
as we run it on a tensor whose maximum value is larger than the first tensor
we gave the model, it will throw an error.

This is a pretty contrived example. But there is a far more insidious example
that can really mess you up:

.. jupyter-execute::

    class BadNetwork2(ter.Module):

        def forward(self, x):
            self.start_forward()
            # here we try to use two different models depending
            # on the mean value of x. What could go wrong?
            if x.mean() > 0:
                return self.make(ter.LazyLinear, 10)(x)
            else:
                return self.make(ter.LazyLinear, 10)(x)

In this case, we want to use different weights in our neural network in a
data-dependant manner. However, this will silently fail! Terrace uses
a counter to determine what module to run when ``make`` is called. In both
the ``if`` and ``else`` clauses of the example, ``make`` will return the
first ``LazyLinear`` module we initialized. This means that we're actually
using the **same weights for all inputs**! The correct way to have the
desired effect is to make sure you initialize both linear layers in the same
order every time:

.. jupyter-execute::

    class FixedNetwork2(ter.Module):

        def forward(self, x):
            self.start_forward()
            
            if_linear = self.make(ter.LazyLinear, 10)
            else_linear = self.make(ter.LazyLinear, 10)

            if x.mean() > 0:
                return if_linear(x)
            else:
                return else_linear(x)

.. warning::

    Be **very** careful using Terrace ``Modules`` with data-dependant
    control flow.

Another, less insidious gothcya is that, since ``Module`` parameters
are initialized lazily, you'll need to make sure to run your module at
least once before calling its ``parameters()``. For instance, you need
to run all your models on a dummy batch before configuring any optimizers
for training.