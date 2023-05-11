CategoricalTensors
===================

.. jupyter-execute::
    :hide-code:
    
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))

.. warning::
    ``CategoricalTensors`` are probably the weirdest things in this library.
    The API is unintuitive and brittle, and therefore will likely be changed
    soon. Use at your own risk.

The nice thing about Terrace ``Module``\s is that they allow us to do
automatic dimension inference of inputs. However, things get complicated
when we start dealing with categorical data. If we want to create an
``nn.Embedding`` layer in one of our modules, we need to know how many
classes we're dealing with. This is where Terrace's ``CategoricalTensor``
comes in.

``CategoricalTensors`` are ``torch.Tensor`` subclasses (always with
``dtype=torch.long``) that have an additional member ``num_classes``.
Usually, ``num_classes`` is an ``int`` that describes the number
of classes the tensor might contain -- that is, it means that all
the items in the tensor are in the range ``[0, num_classes)``.

We can create a ``CategoricalTensor`` like so:

.. jupyter-execute::

    import terrace as ter
    import torch

    ct = ter.CategoricalTensor(torch.zeros((8, 1), dtype=torch.long), num_classes=6)

The nice thing about categorical tensors is that we can now use this metadata
in our ``Modules``. For instance, Terrace's ``LazyEmbedding`` module takes
as input categorical tensors so it knows what size of embeddings to use.
Using categorical tensors with lazy embeddings is the primary use case for
this class.

.. jupyter-execute::

    embed = ter.LazyEmbedding(4)
    print(embed(ct).shape)

Note that we're giving the embedding layer a tensor of shape ``(..., 1)``.
This is what the layer expects in the usual use case where you have a single
categorical feature per "thing" -- for instance, in NLP you might have 
a sequence of tokens which are described by a single number.

However, the categorical tensor and lazy embedding classes also support the
slightly more esoteric use case of multiple categorical features per "thing".
For instance, in a molecule, an atom can have both an element and a formal
charge. We can store all the atomic data for one molecule in a single 
tensor of  shape ``(N, 2)``, where ``N`` is the number of atoms in the
molecule.

.. jupyter-execute::

    N = 8
    mol_data = torch.zeros((N, 2), dtype=torch.long) # a very boring molecule

In this case, ``mol_data[...,0]`` represents the element and ``mol_data[...,1]``
represents the formal charge. However, these two slices can have different
``num_classes``. If there are 4 possible elements and 3 possible formal charges,
how do we make this a categorical tensor? Turns out, just use a tuple:

.. jupyter-execute::

    mol_data = ter.CategoricalTensor(
        torch.zeros((N, 2), dtype=torch.long),
        num_classes=(4, 3)
    )

The nice thing about this is that the ``LazyEmbedding`` layer will handle this
automatically for us.

.. jupyter-execute::

    embed = ter.LazyEmbedding(4)
    print(embed(mol_data).shape)

In the above example, the output embed dim is 8 because the lazy embedding
created **two** embeddings layers each outputting a dimension of 4. The results
from both these layers are concatenated together.


